"""Split full-panel matching and independent per-haplotype focal lookup."""

import dataclasses
import json
import logging
import pathlib

import numpy as np
import pandas as pd
import tsinfer

logger = logging.getLogger(__name__)


def match_ancestors(
    config_path: pathlib.Path,
    output_path: pathlib.Path,
    threads: int,
    cache_size: int | None = None,
) -> None:
    """Match ancestors before samples, preserving the raw reference TS.

    Keeping samples in the configuration preserves their full contig length and
    pre-created individual rows for :func:`match_samples`. Original ancestor
    times must put every ancestor group strictly before the first sample group.
    """
    cfg = tsinfer.config.Config.from_toml(config_path)
    jobs = json.loads(tsinfer.pipeline.compute_groups_json(cfg))
    sample_groups = [job["group"] for job in jobs if job["source"] == "samples"]
    first_sample_group = min(sample_groups)
    ancestor_groups = [job["group"] for job in jobs if job["source"] == "ancestors"]
    if any(group >= first_sample_group for group in ancestor_groups):
        raise ValueError("All ancestor groups must precede the first sample group")
    ts = tsinfer.pipeline.match(
        cfg, group_stop=first_sample_group, num_threads=threads, cache_size=cache_size
    )
    ts.dump(output_path)
    logger.info("Wrote raw ancestor reference to %s", output_path)


def _inferred_focal_dataframe(
    ancestors_path: pathlib.Path, samples_path: pathlib.Path
) -> pd.DataFrame:
    """Relate each non-padding focal position to its panel sample_id string.

    Multi-focal ancestors contribute multiple rows to :func:`find_focal_ancestors`.
    """
    panel = tsinfer.vcz.open_store(ancestors_path)
    samples = tsinfer.vcz.open_store(samples_path)
    ancestor_ids = np.asarray(panel["sample_id"][:].tolist(), dtype=str)
    panel_positions = panel["variant_position"][:]
    sample_positions = samples["variant_position"][:]
    sample_rows = np.searchsorted(sample_positions, panel_positions)
    if np.any(sample_rows == len(sample_positions)):
        raise ValueError("An inferred focal position is absent from the sample store")
    if not np.array_equal(sample_positions[sample_rows], panel_positions):
        raise ValueError("An inferred focal position is absent from the sample store")
    frequencies = samples["variant_match_eval_derived_af"][sample_rows]
    allele_counts = samples["variant_match_eval_derived_ac"][sample_rows]
    position_rows = {int(position): row for row, position in enumerate(panel_positions)}
    records = []
    for ancestor_id, ancestor_time, positions in zip(
        ancestor_ids,
        panel["sample_time"][:],
        panel["sample_focal_positions"][:],
        strict=True,
    ):
        for position in positions:
            if position >= 0:
                row = position_rows[int(position)]
                records.append(
                    {
                        "focal_position": int(position),
                        "inferred_ancestor_id": ancestor_id,
                        "inferred_node_time": float(ancestor_time),
                        "derived_af": float(frequencies[row]),
                        "derived_ac": int(allele_counts[row]),
                    }
                )
    return pd.DataFrame.from_records(records)


def find_focal_ancestors(
    samples_path: pathlib.Path,
    ancestors_path: pathlib.Path,
    output_path: pathlib.Path,
    ancestral_state: dict,
    true_dataframe_path: pathlib.Path | None = None,
    sample_selection: str | None = None,
) -> None:
    """Save derived-focal candidates separately for each sample haplotype.

    The NPZ stores sample_id and ploidy_index in sample-store order, ancestor_id
    in panel-column order, and a ragged array of sorted distinct ancestor_index
    slices delimited by offsets. All arrays load with allow_pickle=False.
    These indices are panel columns, never matched TS node IDs. Focal relations
    at recurrent truth sites describe allele sharing, not mutation origins.
    True singleton-origin rows have no panel ancestor and contribute no candidates;
    multiple eligible sites selecting one truth node share its panel index.
    """
    samples = tsinfer.vcz.open_store(samples_path)
    panel = tsinfer.vcz.open_store(ancestors_path)
    positions = panel["variant_position"][:]
    ancestor_ids = np.asarray(panel["sample_id"][:].tolist(), dtype=str)
    if true_dataframe_path is None:
        focal = _inferred_focal_dataframe(ancestors_path, samples_path)
        id_column = "inferred_ancestor_id"
    else:
        focal = pd.read_csv(
            true_dataframe_path,
            dtype={"inferred_ancestor_id": str, "true_ancestor_id": str},
        )
        focal = focal.loc[~focal.true_mutation_is_singleton]
        id_column = "true_ancestor_id"
    ancestor_columns = {ancestor_id: i for i, ancestor_id in enumerate(ancestor_ids)}
    derived_ac = None
    if true_dataframe_path is None:
        first_relations = focal.drop_duplicates("inferred_ancestor_id", keep="first")
        count_by_id = first_relations.set_index("inferred_ancestor_id")["derived_ac"]
        derived_ac = count_by_id.loc[ancestor_ids].to_numpy(dtype=np.int64)
    position_columns = {}
    for position, ancestor_id in zip(
        focal.focal_position, focal[id_column], strict=True
    ):
        column = ancestor_columns[ancestor_id]
        position_columns.setdefault(int(position), []).append(column)

    sample_positions = samples["variant_position"][:]
    sample_rows = np.searchsorted(sample_positions, positions)
    columns = tsinfer.vcz.resolve_samples_selection(samples, sample_selection)
    genotypes = samples["call_genotype"].oindex[sample_rows, columns, :]
    alleles = samples["variant_allele"][sample_rows]
    ancestral = alleles[:, 0]
    if not ancestral_state.get("is_reference", False):
        ancestral = samples[ancestral_state["field"]][sample_rows]
    # Replace missing indices only for allele lookup; the called mask retains them.
    called = genotypes >= 0
    allele_indices = np.maximum(genotypes, 0)
    called_alleles = np.take_along_axis(alleles[:, None, :], allele_indices, axis=2)
    derived = called & (called_alleles != ancestral[:, None, None])

    sample_ids = np.asarray(samples["sample_id"].oindex[columns].tolist(), dtype=str)
    ploidy = genotypes.shape[2]
    offsets = [0]
    candidate_rows = []
    for sample_index in range(len(sample_ids)):
        for ploidy_index in range(ploidy):
            derived_positions = positions[derived[:, sample_index, ploidy_index]]
            candidates = []
            for position in derived_positions:
                candidates.extend(position_columns.get(int(position), []))
            candidate_array = np.asarray(candidates, dtype=np.int64)
            candidate_array = np.unique(candidate_array)
            candidate_rows.append(candidate_array)
            offsets.append(offsets[-1] + len(candidate_array))
    ancestor_index = np.concatenate(candidate_rows)
    arrays = {
        "sample_id": np.repeat(sample_ids, ploidy),
        "ploidy_index": np.tile(np.arange(ploidy), len(sample_ids)),
        "ancestor_id": ancestor_ids,
        "offsets": np.asarray(offsets, dtype=np.int64),
        "ancestor_index": ancestor_index,
    }
    if derived_ac is not None:
        arrays["derived_ac"] = derived_ac
    np.savez_compressed(output_path, **arrays)
    logger.info(
        "Wrote focal candidates for %d haplotypes to %s", len(offsets) - 1, output_path
    )


def match_samples(
    config_path: pathlib.Path,
    ancestors_ts_path: pathlib.Path,
    output_path: pathlib.Path,
    match_file_path: pathlib.Path,
    threads: int,
    cache_size: int | None = None,
) -> None:
    """Match samples against the unmodified reference from :func:`match_ancestors`.

    Preserve source selection/order and individual metadata so recreated job
    individual IDs correspond to the reference rows. Native JSONL has one record
    per haplotype, keyed by (source, sample_id, ploidy_index); parent IDs are raw
    TS node IDs, joined to panel IDs through node metadata. Focal NPZs do not
    participate in matching.
    """
    cfg = tsinfer.config.Config.from_toml(config_path)
    sample_match = dataclasses.replace(
        cfg.match,
        sources={"samples": cfg.match.sources["samples"]},
        reference_ts=ancestors_ts_path,
        output=output_path,
        path_compression=False,
    )
    sample_cfg = dataclasses.replace(
        cfg,
        sources={"samples": cfg.sources["samples"]},
        ancestors=[],
        match=sample_match,
        post_process=None,
        augment_sites=None,
    )
    ts = tsinfer.pipeline.match(
        sample_cfg,
        num_threads=threads,
        match_file=match_file_path,
        cache_size=cache_size,
    )
    ts.dump(output_path)
    logger.info(
        "Wrote raw samples to %s and matches to %s", output_path, match_file_path
    )
