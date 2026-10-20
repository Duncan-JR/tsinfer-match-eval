"""Save a native linesweep group's complete matching inputs and outputs."""

import argparse
import collections
import dataclasses
import json
import logging
import pathlib
import shutil
import subprocess

import numpy as np
import tomli_w
import tsinfer
import tskit

from . import native_matching

logger = logging.getLogger(__name__)


def write_group_panel(panel, jobs, output):
    """Copy the group's columns, retaining original ancestor IDs and site axis.

    Copy genotypes one physical variant chunk at a time so exporting a group
    does not require decoding the entire ancestor matrix into memory.
    """
    ids = panel["sample_id"][:]
    columns_by_id = {str(sample_id): i for i, sample_id in enumerate(ids)}
    columns = [columns_by_id[job.sample_id] for job in jobs]
    positions = panel["variant_position"][:]
    genotypes = panel["call_genotype"]
    root = tsinfer.vcz.setup_ancestor_zarr(
        len(positions),
        positions,
        panel["variant_allele"][:],
        np.zeros(len(positions), dtype=np.int8),
        panel["sequence_intervals"][:],
        store=output,
        variants_chunk_size=genotypes.chunks[0],
        samples_chunk_size=min(len(columns), genotypes.chunks[1]),
        contig_id=str(panel["contig_id"][0]),
        contig_length=int(panel["contig_length"][0]),
    )
    root["call_genotype"].resize((len(positions), len(columns), 1))
    for start in range(0, len(positions), genotypes.chunks[0]):
        end = min(start + genotypes.chunks[0], len(positions))
        root["call_genotype"][start:end] = genotypes.oindex[start:end, columns, :]
    for field in ("sample_time", "sample_start_position", "sample_end_position"):
        root[field].resize((len(columns),))
        root[field][:] = panel[field].oindex[columns]
    focal_positions = panel["sample_focal_positions"].oindex[columns, :]
    tsinfer.vcz.finalize_ancestor_zarr(root, focal_positions)
    root["sample_id"][:] = ids[columns]


def write_inputs(cfg, config_path, panel, jobs, checkpoint_dir, frequency):
    """Write IDs, haplotypes and a native TOML describing this group's match.

    The replay config uses paths relative to the checkpoint directory, so the
    bundle can be moved and replayed with :func:`replay`.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_group_panel(panel, jobs, checkpoint_dir / "group.zarr")
    records = [dataclasses.asdict(job) for job in jobs]
    checkout = pathlib.Path(tsinfer.__file__).resolve().parent
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "ancestor_match_freq": frequency,
        "group": jobs[0].group,
        "min_frequency": min(job.time for job in jobs),
        "max_frequency": max(job.time for job in jobs),
        "tsinfer_version": tsinfer.__version__,
        "tsinfer_revision": revision,
        "jobs": records,
    }
    (checkpoint_dir / "group.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(config_path, checkpoint_dir / "original_config.toml")
    native = {
        "source": [
            {
                "name": "ancestors",
                "path": "group.zarr",
                "sample_time": "sample_time",
            }
        ],
        "ancestral_state": {"path": "group.zarr", "is_reference": True},
        "match": {
            "reference_ts": "before.trees",
            "output": "after.trees",
            "path_compression": False,
            "sources": {
                "ancestors": dataclasses.asdict(cfg.match.sources["ancestors"]),
            },
        },
    }
    (checkpoint_dir / "group.toml").write_text(tomli_w.dumps(native))


def match_ancestors(
    config_path: pathlib.Path,
    output_path: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    frequency: float,
    threads: int,
    cache_size: int | None = None,
    read_workers: int | None = None,
) -> None:
    """Run native ancestor groups, saving the first with any time <= frequency.

    Native inferred ancestor times are derived counts divided by cohort
    haplotype count. Linesweep groups can mix times: the threshold is applied
    to the minimum time, without splitting or reordering a group. Both tree
    sequences remain raw, with original node IDs and pre-created individuals
    for subsequent sample matching in :mod:`lib.matching`.

    Use the native group processor and one scheduled reader throughout; only
    the two requested snapshots are written, avoiding per-group HPC I/O.
    """
    cfg = tsinfer.config.Config.from_toml(config_path)
    jobs, individuals, populations = tsinfer.pipeline._build_jobs_with_metadata(cfg)
    first_sample_group = min(job.group for job in jobs if job.source == "samples")
    ancestor_jobs = [job for job in jobs if job.source == "ancestors"]
    if any(job.group >= first_sample_group for job in ancestor_jobs):
        raise ValueError("All ancestor groups must precede the first sample group")
    groups = collections.defaultdict(list)
    for index, job in enumerate(ancestor_jobs):
        groups[job.group].append((index, job))
    selected = next((job.group for job in ancestor_jobs if job.time <= frequency), None)
    if selected is None:
        raise ValueError(f"No ancestor group contains a frequency <= {frequency}")

    panel = tsinfer.vcz.open_store(cfg.ancestors[0].path)
    positions = panel["variant_position"][:]
    alleles = panel["variant_allele"][:]
    intervals = panel["sequence_intervals"][:]
    samples = tsinfer.vcz.open_store(cfg.sources["samples"].path)
    schedule = [(job.source, job.sample_id, job.ploidy_index) for job in ancestor_jobs]
    reader = tsinfer.vcz.HaplotypeReader(
        {name: cfg.sources[name] for name in cfg.match.sources},
        positions,
        alleles[:, 0],
        schedule=schedule,
        cache_size_mb=cache_size,
        read_workers=read_workers,
        num_threads=threads,
    )
    try:
        ts = tsinfer.matching.make_root_ts(
            tsinfer.vcz.sequence_length(samples),
            positions,
            intervals,
            max_time=float(np.max(panel["sample_time"][:])),
            individuals=individuals if len(individuals) > 0 else None,
            populations=populations if len(populations) > 0 else None,
            allele_mapper=reader.allele_mapper,
        )
        for group, group_jobs in groups.items():
            logger.info(
                "Matching ancestor group %d (%d haplotypes)", group, len(group_jobs)
            )
            if group == selected:
                write_inputs(
                    cfg,
                    config_path,
                    panel,
                    [job for _, job in group_jobs],
                    checkpoint_dir,
                    frequency,
                )
                ts.dump(checkpoint_dir / "before.trees")
                logger.info(
                    "Saved checkpoint before group %d to %s", group, checkpoint_dir
                )
            ts = tsinfer.pipeline._process_group(
                ts,
                group_idx=group,
                group_jobs=group_jobs,
                positions=positions,
                path_compression=False,
                allele_mapper=reader.allele_mapper,
                reader=reader,
                num_threads=threads,
                match_fh=None,
                config_str=cfg.format(),
                progress=False,
                source_parameters=cfg.match.sources,
            )
            if group == selected:
                ts.dump(checkpoint_dir / "after.trees")
                logger.info("Saved checkpoint after group %d", group)
        ts.dump(output_path)
        logger.info("Wrote complete raw ancestor reference to %s", output_path)
    finally:
        reader.shutdown()


def replay(
    checkpoint_dir: pathlib.Path,
    threads: int | None = None,
    cache_size: int | None = None,
    read_workers: int | None = None,
) -> None:
    """Recreate after.trees using only the movable checkpoint's saved inputs.

    Current tsinfer main does not honour match.reference_ts in its pipeline
    runner. Use its native group processor directly, with the saved original
    jobs and IDs, rather than requiring a patched tsinfer checkout.
    """
    if threads is None:
        threads = tsinfer.config.DEFAULT_NUM_THREADS
    checkpoint_dir = checkpoint_dir.resolve()
    cfg = tsinfer.config.Config.from_toml(checkpoint_dir / "group.toml")
    for source in cfg.sources.values():
        source.path = checkpoint_dir / source.path
    cfg.ancestral_state.path = checkpoint_dir / cfg.ancestral_state.path
    cfg.match.reference_ts = checkpoint_dir / cfg.match.reference_ts
    cfg.match.output = checkpoint_dir / cfg.match.output
    manifest = json.loads((checkpoint_dir / "group.json").read_text())
    jobs = [tsinfer.config.MatchJob(**record) for record in manifest["jobs"]]
    reference = tskit.load(cfg.match.reference_ts)
    ts = native_matching.match_reference(
        cfg, reference, jobs, threads, cache_size, read_workers
    )
    ts.dump(cfg.match.output)
    logger.info("Replayed group %d to %s", manifest["group"], cfg.match.output)


def main():
    """Command-line entry point for replaying a saved checkpoint."""
    parser = argparse.ArgumentParser(description="Replay a native ancestor checkpoint")
    parser.add_argument("checkpoint_dir", type=pathlib.Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cache-size", type=int)
    parser.add_argument("--read-workers", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    replay(args.checkpoint_dir, args.threads, args.cache_size, args.read_workers)


if __name__ == "__main__":
    main()
