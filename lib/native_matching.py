"""Native matching against an existing reference, independent of pipeline resume."""

import collections
import contextlib
import logging
import pathlib

import numpy as np
import tsinfer
import tskit

logger = logging.getLogger(__name__)


def match_reference(
    cfg: tsinfer.config.Config,
    reference: tskit.TreeSequence,
    jobs: list[tsinfer.config.MatchJob],
    threads: int,
    cache_size: int | None = None,
    read_workers: int | None = None,
    match_file_path: pathlib.Path | None = None,
) -> tskit.TreeSequence:
    """Apply native matching groups to a raw reference without recreating it.

    Jobs retain their original ordering, group numbers and metadata row IDs.
    The caller supplies a reference with those individuals/populations already
    present, as produced by the ancestor stage. This uses the same group
    processor as tsinfer's unified pipeline, which currently ignores its
    reference_ts argument on main.
    """
    positions = np.asarray(reference.sites_position, dtype=np.int32)
    ancestral = np.array([site.ancestral_state for site in reference.sites()])
    groups = collections.defaultdict(list)
    for index, job in enumerate(jobs):
        groups[job.group].append((index, job))
    ordered_jobs = sorted(jobs, key=lambda job: (job.group, job.haplotype_index))
    schedule = [(job.source, job.sample_id, job.ploidy_index) for job in ordered_jobs]
    reader = tsinfer.vcz.HaplotypeReader(
        {name: cfg.sources[name] for name in cfg.match.sources},
        positions,
        ancestral,
        schedule=schedule,
        cache_size_mb=cache_size,
        read_workers=read_workers,
        num_threads=threads,
    )
    try:
        with contextlib.ExitStack() as stack:
            match_file = None
            if match_file_path is not None:
                match_file = stack.enter_context(match_file_path.open("w"))
            ts = reference
            for group in sorted(groups):
                ts = tsinfer.pipeline._process_group(
                    ts,
                    group_idx=group,
                    group_jobs=groups[group],
                    positions=positions,
                    path_compression=False,
                    allele_mapper=reader.allele_mapper,
                    reader=reader,
                    num_threads=threads,
                    match_fh=match_file,
                    config_str=cfg.format(),
                    progress=False,
                    source_parameters=cfg.match.sources,
                )
            return ts
    finally:
        reader.shutdown()
