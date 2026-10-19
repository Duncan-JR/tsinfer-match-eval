"""Small helpers for the sample VCZ and native tsinfer configuration."""

import logging
import os
import pathlib
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


def validate_config(config):
    """Validate the scalar workflow interface once, before preparing any data."""
    if config.get("focal_choice", "left") not in ("left", "right"):
        raise ValueError("focal_choice must be left or right")
    for name in (
        "ac_cutoff",
        "min_focal_region_size",
        "ancestor_chunk_size",
        "inference_threads",
        "matching_threads",
        "sample_prefetch_chunks",
        "max_pending_haplotypes",
    ):
        value = config[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
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
    if type(config["path_output"]) is not bool:
        raise ValueError("path_output must be a boolean")
    names = [dataset["name"] for dataset in config["datasets"]]
    if len(names) == 0 or len(set(names)) != len(names):
        raise ValueError("datasets must have distinct names")
    for dataset in config["datasets"]:
        if dataset.get("include") is not None and dataset.get("exclude") is not None:
            raise ValueError("Choose at most one of include and exclude")
