"""Small helpers for the sample VCZ and native tsinfer configuration."""

import logging
import pathlib
import shutil
import subprocess

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
