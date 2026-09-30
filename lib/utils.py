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


def add_singleton_mask(
    source: pathlib.Path, output: pathlib.Path, ancestral_state: dict
) -> None:
    """Copy the VCZ and annotate singletons from its current genotype calls.

    Under the biallelic input contract, code 1 is derived when REF is ancestral;
    otherwise code 0 is derived. Missing calls do not enter allele counts.
    The original genotypes stay intact. Native TOML filtering excludes the mask.
    """
    shutil.copytree(source, output)
    store = zarr.open_group(output, mode="a", use_consolidated=False)
    genotypes = store["call_genotype"][:]
    alleles = store["variant_allele"][:]
    ancestral = alleles[:, 0]
    if not ancestral_state.get("is_reference", False):
        ancestral = store[ancestral_state["field"]][:]
    ancestral_is_reference = ancestral == alleles[:, 0]
    derived_calls = np.where(
        ancestral_is_reference[:, None, None], genotypes == 1, genotypes == 0
    )
    allele_count = derived_calls.sum(axis=(1, 2))
    called_count = (genotypes >= 0).sum(axis=(1, 2))
    frequency = np.divide(
        allele_count,
        called_count,
        out=np.zeros(len(allele_count), dtype=float),
        where=called_count > 0,
    )
    chunks = store["variant_position"].chunks
    mask = store.create_array(
        "variant_match_eval_singleton_mask", data=allele_count == 1, chunks=chunks
    )
    mask.attrs["_ARRAY_DIMENSIONS"] = ["variants"]
    af = store.create_array(
        "variant_match_eval_derived_af", data=frequency, chunks=chunks
    )
    af.attrs["_ARRAY_DIMENSIONS"] = ["variants"]
    zarr.consolidate_metadata(output)
    logger.info("Annotated %d observed singleton sites", int(mask[:].sum()))


def write_inference_config(
    samples: pathlib.Path,
    ancestors: pathlib.Path,
    output: pathlib.Path,
    ancestral_state: dict,
) -> None:
    """Write native inference/matching TOML for the selected ancestor panel.

    Both sources stay in match.sources so ancestor matching obtains the full
    sample contig length. A true-panel configuration is for matching only.
    """
    source = {
        "name": "samples",
        "path": str(samples.resolve()),
        "exclude": "INFO/match_eval_singleton_mask == 1",
    }
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
                "ancestors": {"node_flags": 0, "create_individuals": False},
                "samples": {"node_flags": 1, "create_individuals": True},
            },
        },
    }
    output.write_text(tomli_w.dumps(native))
    logger.info("Wrote %s", output)
