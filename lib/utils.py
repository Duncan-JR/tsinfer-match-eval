"""Small helpers for the sample VCZ and native tsinfer configuration."""

import logging
import os
import pathlib
import re
import shutil
import subprocess

import numcodecs
import numpy as np
import tomli_w
import tsinfer
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


def make_inference_config(
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

    Exclude selected-cohort singletons using the annotated derived counts.
    Both sources stay in match.sources so ancestor matching obtains the full
    sample contig length. A true-panel configuration is for matching only.
    """
    if include is not None:
        include = f"({include}) & (INFO/match_eval_derived_ac != 1)"
    elif exclude is not None:
        exclude = f"({exclude}) | (INFO/match_eval_derived_ac == 1)"
    else:
        exclude = "INFO/match_eval_derived_ac == 1"

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


def configure_helpers(helpers):
    """Apply numerical/codec limits before spawning and inside each child."""
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = str(helpers["blas_threads"])
    numcodecs.blosc.set_nthreads(helpers["blosc_threads"])
    zarr.config.set(
        {
            "async.concurrency": helpers["zarr_async_concurrency"],
            "threading.max_workers": helpers["zarr_threading_max_workers"],
        }
    )


def cpu_allocation(workers, helpers):
    """Reserve match/loader CPUs and the simultaneously active decoding helpers."""
    decoding = helpers["zarr_threading_max_workers"] + helpers["blosc_threads"]
    packing = workers * decoding
    matching = workers + 1 + decoding
    return max(packing, matching) + (workers + 1) * (helpers["blas_threads"] - 1)


def parameter_values(value):
    """Normalise a scalar or explicit parameter grid axis to a list."""
    return value if isinstance(value, list) else [value]


def min_region_size_sites(sites_position, min_region_size_kb):
    """Convert kb to a rounded mean inference-site count, with a minimum of one.

    Use the observed first-to-last site span rather than the retained chromosome
    length. The exact-cover implementation still filters connected unions in
    site units; this is a genome-wide mean conversion, not a local bp filter.
    """
    if len(sites_position) < 2:
        raise ValueError("Converting min_region_size_kb requires at least two sites")
    span_bp = float(sites_position[-1] - sites_position[0])
    sites_per_kb = len(sites_position) * 1000 / span_bp
    minimum = round(min_region_size_kb * sites_per_kb)
    return max(1, minimum)


def validate_config(config):
    """Validate workflow settings and optional parameter grid axes."""
    if "min_focal_region_size" in config:
        raise ValueError(
            "Replace min_focal_region_size with min_region_size_kb; "
            "the configured minimum is now in kb, not inference sites"
        )
    config_name = config["config_name"]
    if (
        not isinstance(config_name, str)
        or re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", config_name) is None
    ):
        raise ValueError(
            "config_name must use lowercase letters, digits and underscores"
        )
    if config.get("focal_choice", "left") not in ("left", "right"):
        raise ValueError("focal_choice must be left or right")
    obsolete = {"af_cutoff", "min_region_size_kb"} & config.keys()
    if obsolete:
        names = ", ".join(sorted(obsolete))
        raise ValueError(f"Move {names} into each datasets entry")
    for dataset in config["datasets"]:
        validate_dataset_parameters(dataset)
    for name in (
        "ancestor_chunk_size",
        "inference_threads",
        "matching_threads",
        "sample_prefetch_chunks",
    ):
        value = config[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    pending = config["max_pending_haplotypes"]
    if pending is not None and (type(pending) is not int or pending < 1):
        raise ValueError("max_pending_haplotypes must be a positive integer or null")
    for name in ("recombination", "mismatch"):
        if not 0 < config["hmm"][name] < 1:
            raise ValueError(f"hmm.{name} must be between zero and one")
    for name in (
        "blas_threads",
        "blosc_threads",
        "zarr_async_concurrency",
        "zarr_threading_max_workers",
    ):
        value = config["helper_threads"][name]
        if type(value) is not int or value < 1:
            raise ValueError(f"helper_threads.{name} must be a positive integer")
    names = [dataset["name"] for dataset in config["datasets"]]
    if len(names) == 0 or len(set(names)) != len(names):
        raise ValueError("datasets must have distinct names")
    for dataset in config["datasets"]:
        if dataset.get("include") is not None and dataset.get("exclude") is not None:
            raise ValueError("Choose at most one of include and exclude")


def validate_dataset_parameters(dataset):
    """Require distinct, valid axes directly on each dataset."""
    for name in ("af_cutoff", "min_region_size_kb"):
        if name not in dataset:
            raise ValueError(f"Dataset {dataset['name']} requires {name}")
        values = parameter_values(dataset[name])
        if len(values) == 0:
            raise ValueError(f"{name} must contain at least one value")
        for value in values:
            if name == "af_cutoff":
                if type(value) not in (int, float) or not 0 <= value <= 1:
                    raise ValueError("af_cutoff values must be numbers between 0 and 1")
            elif type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError(
                    "min_region_size_kb values must be finite nonnegative numbers"
                )
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicates")


def validate_parameter_pair(dataset, af_cutoff, min_region_size_kb):
    """Reject explicit wildcard pairs outside this dataset's Cartesian product."""
    validate_dataset_parameters(dataset)
    if af_cutoff not in parameter_values(
        dataset["af_cutoff"]
    ) or min_region_size_kb not in parameter_values(dataset["min_region_size_kb"]):
        raise ValueError(
            f"Parameter pair AF={af_cutoff}, minimum={min_region_size_kb} kb "
            f"is not configured for dataset {dataset['name']}"
        )
