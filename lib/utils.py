"""Small helpers for the sample VCZ and native tsinfer configuration."""

import logging
import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd
import tomli_w
import tsinfer
import tskit
import zarr

logger = logging.getLogger(__name__)


def setup_log(path: pathlib.Path) -> None:
    """Send this Snakemake job's Python and tsinfer messages to its rule log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=path, level=logging.INFO, force=True)
    checkout = pathlib.Path(tsinfer.__file__).resolve().parent.parent
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    logger.info("tsinfer %s at %s, commit %s", tsinfer.__version__, checkout, commit)


def annotate_derived_counts(
    source: pathlib.Path,
    output: pathlib.Path,
    ancestral_state: dict,
    samples: str | None = None,
) -> None:
    """Copy the VCZ and annotate derived counts for the native selected cohort.

    Retain both original axes. Read selected columns in physical variant chunks;
    missing calls do not enter either allele count or frequency denominator.
    Under the biallelic input contract, non-reference ancestry makes REF derived.
    """
    shutil.copytree(source, output)
    store = zarr.open_group(output, mode="a", use_consolidated=False)
    columns = tsinfer.vcz.resolve_samples_selection(store, samples)
    genotypes = store["call_genotype"]
    allele_count = np.zeros(genotypes.shape[0], dtype=np.int64)
    frequency = np.zeros(genotypes.shape[0], dtype=float)
    for start in range(0, genotypes.shape[0], genotypes.chunks[0]):
        end = min(start + genotypes.chunks[0], genotypes.shape[0])
        calls = genotypes.oindex[start:end, columns, :]
        alleles = store["variant_allele"][start:end]
        ancestral = alleles[:, 0]
        if not ancestral_state.get("is_reference", False):
            ancestral = store[ancestral_state["field"]][start:end]
        reference_ancestral = ancestral == alleles[:, 0]
        derived = np.where(reference_ancestral[:, None, None], calls == 1, calls == 0)
        counts = derived.sum(axis=(1, 2))
        called = (calls >= 0).sum(axis=(1, 2))
        allele_count[start:end] = counts
        np.divide(counts, called, out=frequency[start:end], where=called > 0)
    chunks = store["variant_position"].chunks
    for name, values in (
        ("variant_match_eval_derived_ac", allele_count),
        ("variant_match_eval_derived_af", frequency),
    ):
        array = store.create_array(name, data=values, chunks=chunks, overwrite=True)
        array.attrs["_ARRAY_DIMENSIONS"] = ["variants"]
    zarr.consolidate_metadata(output)
    logger.info(
        "Annotated %d variants for %d individuals", len(allele_count), len(columns)
    )


def write_inference_config(
    samples: pathlib.Path,
    ancestors: pathlib.Path,
    output: pathlib.Path,
    ancestral_state: dict,
    hmm: dict,
    include: str | None = None,
    exclude: str | None = None,
    sample_selection: str | None = None,
) -> None:
    """Write native inference/matching TOML for the selected ancestor panel.

    Both sources stay in match.sources so ancestor matching obtains the full
    sample contig length. A true-panel configuration is for matching only.
    """
    source = {
        "name": "samples",
        "path": str(samples.resolve()),
    }
    for name, value in (
        ("include", include),
        ("exclude", exclude),
        ("samples", sample_selection),
    ):
        if value is not None:
            source[name] = value
    annotation = {"path": str(samples.resolve()), **ancestral_state}
    panel = {
        "name": "ancestors",
        "path": str(ancestors.resolve()),
        "sources": ["samples"],
    }
    matched = ancestors.with_suffix(".trees")
    native = {
        "source": [source],
        "ancestral_state": annotation,
        "ancestors": [panel],
        "match": {
            "output": str(matched.resolve()),
            "path_compression": False,
            "sources": {
                "ancestors": {
                    "node_flags": 0,
                    "create_individuals": False,
                    **hmm,
                },
                "samples": {
                    "node_flags": 1,
                    "create_individuals": True,
                    **hmm,
                },
            },
        },
    }
    output.write_text(tomli_w.dumps(native))
    logger.info("Wrote %s", output)


def extract_ts_populations(
    samples_path: pathlib.Path, ts_path: pathlib.Path, samples: str | None = None
) -> pd.DataFrame:
    """Join selected Zarr genomes to original nodes via map_to_vcf_model()."""
    store = tsinfer.vcz.open_store(samples_path)
    columns = tsinfer.vcz.resolve_samples_selection(store, samples)
    ids = store["sample_id"].oindex[columns]
    ploidy = store["call_genotype"].shape[2]
    ts = tskit.load(ts_path)
    mapping = ts.map_to_vcf_model()
    nodes_by_id = dict(
        zip(mapping.individuals_name, mapping.individuals_nodes, strict=True)
    )
    if len(nodes_by_id) != len(mapping.individuals_name):
        raise ValueError("Original TS individual names are not unique")
    records = []
    for sample_id in ids:
        if sample_id not in nodes_by_id:
            raise ValueError(f"Selected sample {sample_id} is absent from original TS")
        nodes = nodes_by_id[sample_id]
        if len(nodes) != ploidy or np.any(nodes < 0):
            raise ValueError(
                f"Original TS ploidy does not match selected sample {sample_id}"
            )
        for ploidy_index, node_id in enumerate(nodes):
            node = ts.node(int(node_id))
            if node.individual == tskit.NULL or node.population == tskit.NULL:
                raise ValueError(
                    f"Original node {node.id} has no individual or population"
                )
            label = ts.population(node.population).metadata.get("name")
            if label is None or label == "":
                raise ValueError(f"Original population {node.population} has no name")
            records.append(
                {
                    "source": "samples",
                    "sample_id": str(sample_id),
                    "ploidy_index": ploidy_index,
                    "truth_node_id": node.id,
                    "truth_individual_id": node.individual,
                    "population_id": node.population,
                    "population": label,
                }
            )
    dataframe = pd.DataFrame(
        records,
        columns=[
            "source",
            "sample_id",
            "ploidy_index",
            "truth_node_id",
            "truth_individual_id",
            "population_id",
            "population",
        ],
    )
    return dataframe.set_index(
        ["source", "sample_id", "ploidy_index"], verify_integrity=True
    )


def extract_csv_populations(
    samples_path: pathlib.Path,
    csv_path: pathlib.Path,
    zarr_id_field: str,
    csv_id_field: str,
    pop_field: str,
    samples: str | None = None,
) -> pd.DataFrame:
    """Join selected sample-dimensioned identifiers to explicit CSV labels."""
    store = tsinfer.vcz.open_store(samples_path)
    columns = tsinfer.vcz.resolve_samples_selection(store, samples)
    ids = store["sample_id"].oindex[columns]
    lookup = store[zarr_id_field]
    if lookup.shape != store["sample_id"].shape:
        raise ValueError(f"{zarr_id_field} must have the sample dimension")
    lookup_ids = np.asarray(lookup.oindex[columns].tolist(), dtype=str)
    pedigree = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if (pedigree[csv_id_field] == "").any():
        raise ValueError("CSV identifiers must not be empty")
    labels = pedigree.set_index(csv_id_field, verify_integrity=True)[pop_field]
    selected = labels.reindex(lookup_ids)
    if selected.isna().any() or (selected == "").any():
        raise ValueError(
            "Selected samples have missing CSV identifiers or population labels"
        )
    ploidy = store["call_genotype"].shape[2]
    dataframe = pd.DataFrame(
        {
            "source": "samples",
            "sample_id": np.repeat(np.asarray(ids.tolist(), dtype=str), ploidy),
            "ploidy_index": np.tile(np.arange(ploidy), len(ids)),
            "population": np.repeat(selected.to_numpy(), ploidy),
        }
    )
    return dataframe.set_index(
        ["source", "sample_id", "ploidy_index"], verify_integrity=True
    )


def read_population_metadata(
    samples_path: pathlib.Path, dataset: dict
) -> pd.DataFrame | None:
    """Read optional TS or CSV population enrichment for the configured cohort."""
    source = dataset.get("metadata_source")
    samples = dataset.get("samples")
    if source is None:
        return None
    if source == "ts":
        if dataset.get("ts_path") is None:
            raise ValueError("metadata_source: ts requires ts_path")
        return extract_ts_populations(
            samples_path, pathlib.Path(dataset["ts_path"]).expanduser(), samples
        )
    if source == "csv":
        fields = ("csv_path", "zarr_id_field", "csv_id_field", "pop_field")
        if any(dataset.get(field) is None for field in fields):
            raise ValueError("metadata_source: csv requires " + ", ".join(fields))
        return extract_csv_populations(
            samples_path,
            pathlib.Path(dataset["csv_path"]).expanduser(),
            dataset["zarr_id_field"],
            dataset["csv_id_field"],
            dataset["pop_field"],
            samples,
        )
    raise ValueError(f"Unknown metadata_source: {source}")
