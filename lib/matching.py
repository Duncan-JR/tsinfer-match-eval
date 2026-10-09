"""Split full-panel ancestor and sample matching."""

import dataclasses
import json
import logging
import pathlib

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
