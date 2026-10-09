"""Relate inference sites to unique, older truth-node haplotypes."""

import logging
import pathlib

import numpy as np
import pandas as pd
import tsinfer
import tskit

logger = logging.getLogger(__name__)


def infer_ancestors(config_path: pathlib.Path, threads: int) -> None:
    """Run the current tsinfer ancestor inference with missing-call encoding."""
    native = tsinfer.config.Config.from_toml(config_path)
    panel = native.ancestors[0]
    sources = [native.sources[name] for name in panel.sources]
    tsinfer.ancestors.infer_ancestors(
        sources, panel, native.ancestral_state, num_threads=threads
    )


def primary_mutation(
    site: tskit.Site, ts: tskit.TreeSequence, variant: tskit.Variant
) -> tskit.Mutation | None:
    """Select the event covering the most present-day derived carriers."""
    if len(site.mutations) == 1:
        return site.mutations[0]
    variant.decode(site.id)
    derived = next(
        allele
        for allele in variant.alleles
        if allele is not None and allele != site.ancestral_state
    )
    carriers = variant.samples[variant.genotypes == variant.alleles.index(derived)]
    if len(carriers) == 0:
        logger.warning(
            "No derived carriers at truth site %d (%s)", site.id, site.position
        )
        return None
    tree = ts.at(site.position, tracked_samples=carriers)
    candidates = [m for m in site.mutations if m.derived_state == derived]
    return max(candidates, key=lambda m: tree.num_tracked_samples(m.node))


def extract_true_ancestors(
    inferred_path: pathlib.Path,
    samples_path: pathlib.Path,
    truth_path: pathlib.Path,
    output: pathlib.Path,
    dataframe_path: pathlib.Path,
    chunk_size: int,
) -> None:
    """Select one derived event per site and decode each eligible truth node once.

    The CSV retains every inference site. Rows selecting the same older node
    share a panel sample_id and contribute multiple focal positions to its column.
    Zero-time selections have true_mutation_is_singleton=True and no ancestor ID;
    their sites remain on the matching axis for sample-stage mutations. This flag
    describes the selected terminal origin, including at recurrent sites.

    true_node_id refers to the original truth TS. Haplotypes retain final states,
    missing calls, and original node times. This extraction is available only
    through explicit true-panel targets, outside inferred focal generation.
    """
    inferred = tsinfer.vcz.open_store(inferred_path)
    samples = tsinfer.vcz.open_store(samples_path)
    ts = tskit.load(truth_path)
    positions = inferred["variant_position"][:]
    alleles = inferred["variant_allele"][:]
    true_site_ids = np.searchsorted(ts.sites_position, positions)
    sample_site_ids = np.searchsorted(samples["variant_position"][:], positions)
    frequencies = samples["variant_match_eval_derived_af"][sample_site_ids]
    allele_counts = samples["variant_match_eval_derived_ac"][sample_site_ids]
    inferred_ids = inferred["sample_id"][:]
    inferred_times = inferred["sample_time"][:]
    focal_columns = {}
    for column, focal_positions in enumerate(inferred["sample_focal_positions"][:]):
        for position in focal_positions:
            if position >= 0:
                focal_columns[int(position)] = column

    variant = tskit.Variant(ts, isolated_as_missing=False)
    records = []
    for inference_site_id, true_site_id in enumerate(true_site_ids):
        site = ts.site(int(true_site_id))
        num_mutations = len(site.mutations)
        if num_mutations == 1:
            mutation = site.mutations[0]
        else:
            mutation = primary_mutation(site, ts, variant)
        if mutation is None:
            raise ValueError(
                f"No derived carriers at truth site {site.id} ({site.position})"
            )
        column = focal_columns[int(site.position)]
        node_time = float(ts.nodes_time[mutation.node])
        records.append(
            {
                "inference_site_id": inference_site_id,
                "true_site_id": site.id,
                "focal_position": int(site.position),
                "inferred_ancestor_id": str(inferred_ids[column]),
                "true_ancestor_id": None,
                "true_mutation_id": mutation.id,
                "true_mutation_is_singleton": node_time == 0,
                "true_node_id": mutation.node,
                "true_node_time": node_time,
                "inferred_node_time": float(inferred_times[column]),
                "derived_af": float(frequencies[inference_site_id]),
                "derived_ac": int(allele_counts[inference_site_id]),
                "num_mutations": num_mutations,
            }
        )
    records.sort(key=lambda row: (-row["true_node_time"], row["inference_site_id"]))
    node_columns = {}
    ancestor_nodes = []
    focal_positions = []
    for row in records:
        if row["true_mutation_is_singleton"]:
            continue
        node_id = row["true_node_id"]
        column = node_columns.get(node_id)
        if column is None:
            column = len(ancestor_nodes)
            node_columns[node_id] = column
            ancestor_nodes.append(node_id)
            focal_positions.append([])
        row["true_ancestor_id"] = f"a{column}"
        focal_positions[column].append(row["focal_position"])
    ancestor_nodes = np.asarray(ancestor_nodes, dtype=np.int32)
    dataframe = pd.DataFrame.from_records(records)
    logger.info(
        "Selected %d site records and %d unique older ancestors; %d singleton origins",
        len(dataframe),
        len(ancestor_nodes),
        int(dataframe.true_mutation_is_singleton.sum()),
    )

    # Flag internal nodes in a private copy so tskit can decode their haplotypes.
    tables = ts.dump_tables()
    tables.nodes.flags |= tskit.NODE_IS_SAMPLE
    expanded_ts = tables.tree_sequence()
    root = tsinfer.vcz.setup_ancestor_zarr(
        len(positions),
        positions,
        alleles,
        np.zeros(len(positions), dtype=np.int8),
        inferred["sequence_intervals"][:],
        store=output,
        samples_chunk_size=chunk_size,
        contig_id=str(inferred["contig_id"][0]),
        contig_length=int(inferred["contig_length"][0]),
    )
    num_ancestors = len(ancestor_nodes)
    root["call_genotype"].resize((len(positions), num_ancestors, 1))
    for field in ("sample_time", "sample_start_position", "sample_end_position"):
        root[field].resize((num_ancestors,))
    root["sample_time"][:] = ts.nodes_time[ancestor_nodes]

    for start in range(0, num_ancestors, chunk_size):
        end = min(start + chunk_size, num_ancestors)
        nodes = ancestor_nodes[start:end]
        genotypes = expanded_ts.genotype_matrix(samples=nodes)
        haplotypes = genotypes[true_site_ids].astype(np.int8)
        called = haplotypes >= 0
        if not np.all(called.any(axis=0)):
            raise ValueError("An extracted ancestor has no nonmissing inference sites")
        first = np.argmax(called, axis=0)
        last = len(positions) - 1 - np.argmax(called[::-1], axis=0)
        root["call_genotype"][:, start:end, 0] = haplotypes
        root["sample_start_position"][start:end] = positions[first]
        root["sample_end_position"][start:end] = positions[last] + 1
        logger.info("Wrote true ancestor columns %d:%d", start, end)
    tsinfer.vcz.finalize_ancestor_zarr(root, focal_positions)
    dataframe.to_csv(dataframe_path, index=False)
