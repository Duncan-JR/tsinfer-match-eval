"""Small helpers for the sample VCZ and native tsinfer configuration."""

import importlib.metadata
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import urllib.parse

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
    for name in ["tsinfer", "tsls"]:
        logger.info(
            "%s %s: %s", name, importlib.metadata.version(name), package_revision(name)
        )


def package_revision(name):
    """Identify the installed source, including wheels built from pinned Git."""
    distribution = importlib.metadata.distribution(name)
    source = json.loads(distribution.read_text("direct_url.json"))
    if "vcs_info" in source:
        return {"url": source["url"], "commit": source["vcs_info"]["commit_id"]}
    checkout = pathlib.Path(
        urllib.parse.unquote(urllib.parse.urlparse(source["url"]).path)
    )
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain"], text=True
    )
    return {"url": source["url"], "commit": commit, "status": status.splitlines()}


def annotate_derived_counts(
    source: pathlib.Path,
    output: pathlib.Path,
    ancestral_state: dict,
    samples: str | None = None,
) -> None:
    """Annotate the selected cohort in a VCZ view of the source arrays.

    Retain both original axes. Read selected columns within physical tiles;
    missing calls do not enter either allele count or frequency denominator.
    Under the biallelic input contract, non-reference ancestry makes REF derived.
    """
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    source = source.resolve()
    store = zarr.open_group(source, mode="r", use_consolidated=False)
    columns = tsinfer.vcz.resolve_samples_selection(store, samples)
    genotypes = store["call_genotype"]
    sample_blocks = columns // genotypes.chunks[1]
    selected_chunks = {}
    for block in np.unique(sample_blocks):
        selected = columns[sample_blocks == block]
        selected_chunks[int(block)] = selected - block * genotypes.chunks[1]
    allele_count = np.zeros(genotypes.shape[0], dtype=np.int64)
    frequency = np.zeros(genotypes.shape[0], dtype=float)
    for start in range(0, genotypes.shape[0], genotypes.chunks[0]):
        end = min(start + genotypes.chunks[0], genotypes.shape[0])
        alleles = store["variant_allele"][start:end]
        ancestral = alleles[:, 0]
        if not ancestral_state.get("is_reference", False):
            ancestral = store[ancestral_state["field"]][start:end]
        reference_ancestral = ancestral == alleles[:, 0]
        derived_code = np.where(reference_ancestral, 1, 0)
        counts = np.zeros(end - start, dtype=np.int64)
        called = np.zeros(end - start, dtype=np.int64)
        variant_block = start // genotypes.chunks[0]
        for sample_block, local_columns in selected_chunks.items():
            tile = genotypes.blocks[variant_block, sample_block, :]
            calls = tile[:, local_columns, :]
            del tile
            derived = calls == derived_code[:, None, None]
            counts += derived.sum(axis=(1, 2))
            called += (calls >= 0).sum(axis=(1, 2))
            del calls, derived
        allele_count[start:end] = counts
        np.divide(counts, called, out=frequency[start:end], where=called > 0)
    chunks = store["variant_position"].chunks
    annotations = {
        "variant_match_eval_derived_ac": allele_count,
        "variant_match_eval_derived_af": frequency,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        view = pathlib.Path(staging) / output.name
        view.mkdir()
        for entry in source.iterdir():
            if entry.name in annotations:
                continue
            target = view / entry.name
            if entry.is_dir():
                target.symlink_to(entry, target_is_directory=True)
            else:
                # Consolidation writes root metadata, which belongs to this view.
                shutil.copy2(entry, target)
        annotated = zarr.open_group(view, mode="a", use_consolidated=False)
        for name, values in annotations.items():
            array = annotated.create_array(name, data=values, chunks=chunks)
            array.attrs["_ARRAY_DIMENSIONS"] = ["variants"]
        zarr.consolidate_metadata(view)
        view.rename(output)
    logger.info(
        "Annotated %d variants for %d individuals", len(allele_count), len(columns)
    )


def sample_hmm_settings(samples, hmm, include=None, exclude=None, selection=None):
    """Resolve epsilon on source sites before singleton exclusion, and fixed n.

    Epsilon is per haploid base; divide its expected count over the explicit
    calibration span by the number of retained input variant sites. Both
    sample matchers and ancestor matching use the selected haploid cohort size.
    """
    settings = {name: value for name, value in hmm.items() if value is not None}
    store = tsinfer.vcz.open_store(samples)
    columns = tsinfer.vcz.resolve_samples_selection(store, selection)
    settings["sample_size"] = len(columns) * store["call_genotype"].shape[2]
    epsilon = settings.pop("mismatch_epsilon", None)
    span = settings.pop("mismatch_span_bp", None)
    if epsilon is not None:
        source = tsinfer.config.Source(
            name="samples",
            path=samples,
            include=include,
            exclude=exclude,
            samples=selection,
        )
        resolved = tsinfer.vcz.resolve_sources([source], None, None)
        variants = tsinfer.vcz.select_source_variants(store, resolved.sources[0])
        num_sites = len(variants.positions)
        if num_sites == 0:
            raise ValueError("Epsilon calibration requires retained input sites")
        settings["mismatch"] = epsilon * span / num_sites
        if not 0 < settings["mismatch"] < 1:
            raise ValueError("Calibrated mismatch probability must be between 0 and 1")
        logger.info(
            "Sample epsilon=%g, span=%g bp, source sites=%d, mismatch=%g, n=%d",
            epsilon,
            span,
            num_sites,
            settings["mismatch"],
            settings["sample_size"],
        )
    return settings


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
    Ancestor matching retains native stock probabilities and no map.
    Both sources stay in match.sources so ancestor matching obtains the full
    sample contig length. A true-panel configuration is for matching only.
    """
    hmm = sample_hmm_settings(samples, hmm, include, exclude, sample_selection)
    rate = hmm.get("recombination_rate")
    if isinstance(rate, (str, pathlib.Path)):
        source_map = pathlib.Path(rate).expanduser()
        native_map = output.with_suffix(".recombination_map.txt").resolve()
        map_data = read_recombination_map(source_map)
        with source_map.open() as source:
            next(source)
            chromosome = next(source).split()[0]
        np.savetxt(
            native_map,
            map_data,
            fmt=f"{chromosome}\t%.17g\t%.17g\t%.17g",
            header="Chromosome Position(bp) Rate(cM/Mb) Map(cM)",
            comments="",
        )
        hmm["recombination_rate"] = str(native_map)
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
                    "sample_size": hmm["sample_size"],
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


def read_recombination_map(path):
    """Read position, rate and cumulative cM from a three/four-column HapMap file."""
    path = pathlib.Path(path).expanduser()
    with path.open() as source:
        columns = len(source.readline().split())
    usecols = (1, 2) if columns == 3 else (1, 2, 3)
    data = np.loadtxt(path, skiprows=1, usecols=usecols, ndmin=2)
    if columns == 3:
        distance_cm = np.diff(data[:, 0]) * data[:-1, 1] / 1e6
        cumulative_cm = np.insert(np.cumsum(distance_cm), 0, 0)
        data = np.column_stack((data, cumulative_cm))
    return data


def configure_helpers(helpers):
    """Apply numerical and codec limits before starting matching threads."""
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


def cpu_allocation(workers, helpers, sample_reader_threads=1):
    """Reserve match/loader CPUs and the simultaneously active decoding helpers."""
    decoding = helpers["zarr_threading_max_workers"] + helpers["blosc_threads"]
    # Zarr's async limit controls outstanding I/O, not an extra CPU pool.
    # Its thread executor and codec helpers are shared by the outer readers.
    readers = sample_reader_threads if sample_reader_threads > 1 else 0
    # The coordinator and loader can run while native worker calls release the GIL.
    matching = workers + 2 + readers + decoding
    numerical_threads = workers + 2 + readers
    return matching + numerical_threads * (helpers["blas_threads"] - 1)


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
    if type(config.get("condition_gaps", True)) is not bool:
        raise ValueError("condition_gaps must be true or false")
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
    readers = config.get("sample_reader_threads", 1)
    if type(readers) is not int or readers < 1:
        raise ValueError("sample_reader_threads must be a positive integer")
    pending = config["max_pending_haplotypes"]
    if pending is not None and (type(pending) is not int or pending < 1):
        raise ValueError("max_pending_haplotypes must be a positive integer or null")
    hmm = config["hmm"]
    recombination = hmm.get("recombination")
    rate = hmm.get("recombination_rate")
    if (recombination is None) == (rate is None):
        raise ValueError(
            "Choose exactly one of hmm.recombination and hmm.recombination_rate"
        )
    if recombination is not None and not 0 < recombination < 1:
        raise ValueError("hmm.recombination must be between zero and one")
    if isinstance(rate, (str, pathlib.Path)):
        if not pathlib.Path(rate).expanduser().is_file():
            raise ValueError(f"hmm.recombination_rate map does not exist: {rate}")
    elif rate is not None:
        if not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate < 0:
            raise ValueError(
                "hmm.recombination_rate must be a nonnegative rate or map path"
            )
    mismatch = hmm.get("mismatch")
    epsilon = hmm.get("mismatch_epsilon")
    if (mismatch is None) == (epsilon is None):
        raise ValueError("Choose exactly one of hmm.mismatch and hmm.mismatch_epsilon")
    if mismatch is not None and not 0 < mismatch < 1:
        raise ValueError("hmm.mismatch must be between zero and one")
    if epsilon is not None:
        if not 0 < epsilon < 1:
            raise ValueError("hmm.mismatch_epsilon must be between zero and one")
        span = hmm.get("mismatch_span_bp")
        if span is None or not np.isfinite(span) or span <= 0:
            raise ValueError("hmm.mismatch_span_bp must be positive with epsilon")
    elif hmm.get("mismatch_span_bp") is not None:
        raise ValueError("hmm.mismatch_span_bp requires hmm.mismatch_epsilon")
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
