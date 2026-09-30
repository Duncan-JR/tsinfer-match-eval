"""Evaluate sample copying from allele-count eligible focal ancestors."""

import dataclasses
import json
import logging
import math
import pathlib

import numpy as np
import pandas as pd
import tsinfer
import tskit

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvaluationContext:
    """Arrays and scoring values shared by all sample haplotypes."""

    cutoffs: np.ndarray
    ancestor_first_cutoff: np.ndarray
    reference_parent_column: np.ndarray
    sites_position: np.ndarray
    sequence_intervals: np.ndarray
    sequence_length: float
    num_reference_nodes: int
    log_mismatch_penalty: float
    log_switch_penalty: float


@dataclasses.dataclass
class HaplotypeStats:
    """Scalar path metrics and cutoff vectors for one sample haplotype."""

    evaluated_bp: float
    evaluated_sites: int
    num_switches: int
    num_mismatches: int
    path_log_likelihood: float
    covered_bp: np.ndarray
    covered_sites: np.ndarray
    num_focal_ancestors: np.ndarray


def _build_context(
    reference_ts: tskit.TreeSequence,
    ancestor_ids: np.ndarray,
    derived_ac: np.ndarray,
    ac_cutoff: list[int],
    sample_match_config: tsinfer.config.MatchSourceConfig,
) -> EvaluationContext:
    """Validate shared inputs and precompute ancestor eligibility and scoring."""
    if len(ac_cutoff) == 0:
        raise ValueError("ac_cutoff must be a nonempty list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ac_cutoff):
        raise ValueError("ac_cutoff values must be integers, not booleans")
    cutoffs = np.asarray(ac_cutoff, dtype=np.int64)
    if np.any(cutoffs <= 0) or np.any(np.diff(cutoffs) <= 0):
        raise ValueError("ac_cutoff must contain strictly increasing positive integers")

    if len(ancestor_ids) != len(derived_ac):
        raise ValueError(
            "Focal NPZ ancestor_id and derived_ac arrays are misaligned; rebuild it"
        )
    if ancestor_ids.ndim != 1 or derived_ac.ndim != 1:
        raise ValueError("Focal NPZ ancestor arrays must be one-dimensional; rebuild it")
    if len(np.unique(ancestor_ids)) != len(ancestor_ids):
        raise ValueError("Focal NPZ ancestor_id values are not unique; rebuild it")
    if not np.issubdtype(derived_ac.dtype, np.integer):
        raise ValueError("Focal NPZ derived_ac must be an integer array; rebuild it")
    if np.any(derived_ac <= 0):
        raise ValueError("Focal NPZ derived_ac values must be positive; rebuild it")
    ancestor_first_cutoff = np.searchsorted(cutoffs, derived_ac, side="left")

    ancestor_columns = {
        ancestor_id: column for column, ancestor_id in enumerate(ancestor_ids)
    }
    reference_parent_column = np.full(reference_ts.num_nodes, -1, dtype=np.int64)
    joined_ids = set()
    for node in reference_ts.nodes():
        metadata = node.metadata
        if metadata.get("source") != "ancestors":
            continue
        ancestor_id = str(metadata["sample_id"])
        ploidy_index = metadata["ploidy_index"]
        if ploidy_index != 0 or ancestor_id not in ancestor_columns:
            message = (
                f"Reference ancestor node {node.id} does not match the focal NPZ; "
                "rebuild matching products"
            )
            raise ValueError(message)
        if ancestor_id in joined_ids:
            message = (
                f"Reference has duplicate ancestor metadata for {ancestor_id}; "
                "rebuild it"
            )
            raise ValueError(message)
        joined_ids.add(ancestor_id)
        reference_parent_column[node.id] = ancestor_columns[ancestor_id]
    missing_ids = set(ancestor_ids) - joined_ids
    if missing_ids:
        missing_id = sorted(missing_ids)[0]
        message = (
            f"Focal NPZ ancestor {missing_id} is absent from the reference; "
            "rebuild matching products"
        )
        raise ValueError(message)

    sites_position = reference_ts.sites_position
    if len(sites_position) > 1 and np.any(np.diff(sites_position) <= 0):
        raise ValueError("Reference site positions must be strictly increasing")
    metadata = reference_ts.metadata
    if "sequence_intervals" not in metadata:
        raise ValueError("Reference TS has no sequence_intervals metadata; rebuild it")
    sequence_intervals = np.asarray(metadata["sequence_intervals"], dtype=float)
    if sequence_intervals.ndim != 2 or sequence_intervals.shape[1] != 2:
        raise ValueError(
            "Reference sequence_intervals metadata must contain [left, right] pairs"
        )
    if len(sequence_intervals) > 0:
        interval_lengths = sequence_intervals[:, 1] - sequence_intervals[:, 0]
        if np.any(interval_lengths <= 0):
            raise ValueError("Reference sequence intervals must have positive lengths")
        if np.any(sequence_intervals[1:, 0] < sequence_intervals[:-1, 1]):
            raise ValueError("Reference sequence intervals must be ordered and disjoint")
        if sequence_intervals[0, 0] < 0:
            raise ValueError("Reference sequence intervals must lie within the sequence")
        if sequence_intervals[-1, 1] > reference_ts.sequence_length:
            raise ValueError("Reference sequence intervals must lie within the sequence")

    mismatch = sample_match_config.mismatch
    recombination = sample_match_config.recombination
    if mismatch is None or recombination is None:
        raise ValueError(
            "Sample matching probabilities are absent from the TOML; rebuild the "
            "config and matches"
        )
    if not 0 < mismatch < 1:
        raise ValueError(
            "Sample matching mismatch probability must be between zero and one"
        )
    if not 0 < recombination < 1:
        raise ValueError(
            "Sample matching recombination probability must be between zero and one"
        )
    num_reference_nodes = reference_ts.num_nodes
    log_mismatch_penalty = math.log(mismatch) - math.log1p(-mismatch)
    switch_baseline = -recombination + recombination / num_reference_nodes
    log_switch_penalty = (
        math.log(recombination)
        - math.log(num_reference_nodes)
        - math.log1p(switch_baseline)
    )
    return EvaluationContext(
        cutoffs=cutoffs,
        ancestor_first_cutoff=ancestor_first_cutoff,
        reference_parent_column=reference_parent_column,
        sites_position=sites_position,
        sequence_intervals=sequence_intervals,
        sequence_length=reference_ts.sequence_length,
        num_reference_nodes=num_reference_nodes,
        log_mismatch_penalty=log_mismatch_penalty,
        log_switch_penalty=log_switch_penalty,
    )


def _compute_haplotype_stats(
    record: dict, candidate_columns: np.ndarray, context: EvaluationContext
) -> HaplotypeStats:
    """Compute every cutoff for one native match record in one path traversal."""
    num_cutoffs = len(context.cutoffs)
    candidate_buckets = np.zeros(num_cutoffs, dtype=np.int64)
    candidate_bins = context.ancestor_first_cutoff[candidate_columns]
    eligible = candidate_bins < num_cutoffs
    np.add.at(candidate_buckets, candidate_bins[eligible], 1)
    column_to_bin = {
        int(column): int(first_cutoff)
        for column, first_cutoff in zip(
            candidate_columns[eligible], candidate_bins[eligible], strict=True
        )
    }

    path = sorted(record["path"], key=lambda segment: segment["left"])
    for segment in path:
        left = float(segment["left"])
        right = float(segment["right"])
        parent = int(segment["parent"])
        if left < 0 or right > context.sequence_length or left >= right:
            raise ValueError(f"Invalid path segment in match record: {segment}")
        if parent < 0 or parent >= context.num_reference_nodes:
            raise ValueError(f"Path parent {parent} is outside the reference TS")
    for previous, current in zip(path, path[1:], strict=False):
        if float(previous["right"]) != float(current["left"]):
            raise ValueError("Native path segments are not contiguous")
        if int(previous["parent"]) == int(current["parent"]):
            raise ValueError("Native consecutive path segments have the same parent")

    num_switches = max(len(record["path"]) - 1, 0)
    parent_changes = sum(
        int(previous["parent"]) != int(current["parent"])
        for previous, current in zip(path, path[1:], strict=False)
    )
    if num_switches != parent_changes:
        raise ValueError("Native path segment count does not equal its parent changes")
    num_mismatches = len(record["mutations"])
    path_log_likelihood = (
        num_mismatches * context.log_mismatch_penalty
        + num_switches * context.log_switch_penalty
    )

    covered_bp_buckets = np.zeros(num_cutoffs, dtype=np.float64)
    covered_site_buckets = np.zeros(num_cutoffs, dtype=np.int64)
    evaluated_bp = 0.0
    evaluated_sites = 0
    interval_index = 0
    for segment in path:
        segment_left = float(segment["left"])
        segment_right = float(segment["right"])
        while (
            interval_index < len(context.sequence_intervals)
            and context.sequence_intervals[interval_index, 1] <= segment_left
        ):
            interval_index += 1
        current_interval = interval_index
        while current_interval < len(context.sequence_intervals):
            interval_left, interval_right = context.sequence_intervals[current_interval]
            if interval_left >= segment_right:
                break
            fragment_left = max(segment_left, interval_left)
            fragment_right = min(segment_right, interval_right)
            if fragment_left < fragment_right:
                fragment_bp = fragment_right - fragment_left
                left_site = np.searchsorted(
                    context.sites_position, fragment_left, side="left"
                )
                right_site = np.searchsorted(
                    context.sites_position, fragment_right, side="left"
                )
                fragment_sites = int(right_site - left_site)
                evaluated_bp += fragment_bp
                evaluated_sites += fragment_sites
                parent = int(segment["parent"])
                column = int(context.reference_parent_column[parent])
                first_cutoff = column_to_bin.get(column)
                if first_cutoff is not None:
                    covered_bp_buckets[first_cutoff] += fragment_bp
                    covered_site_buckets[first_cutoff] += fragment_sites
            if interval_right >= segment_right:
                break
            current_interval += 1

    return HaplotypeStats(
        evaluated_bp=evaluated_bp,
        evaluated_sites=evaluated_sites,
        num_switches=num_switches,
        num_mismatches=num_mismatches,
        path_log_likelihood=path_log_likelihood,
        covered_bp=np.cumsum(covered_bp_buckets),
        covered_sites=np.cumsum(covered_site_buckets),
        num_focal_ancestors=np.cumsum(candidate_buckets),
    )


def compute_focal_ancestor_stats(
    focal_ancestors_path: pathlib.Path,
    ancestors_ts_path: pathlib.Path,
    match_file_path: pathlib.Path,
    config_path: pathlib.Path,
    ac_cutoff: list[int],
) -> pd.DataFrame:
    """Return inferred focal-ancestor coverage and native path metrics."""
    with np.load(focal_ancestors_path, allow_pickle=False) as focal:
        required = {
            "sample_id",
            "ploidy_index",
            "ancestor_id",
            "offsets",
            "ancestor_index",
            "derived_ac",
        }
        missing = required - set(focal.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"Focal NPZ is missing {names}; rebuild inferred focal ancestors"
            )
        sample_ids = np.asarray(focal["sample_id"], dtype=str)
        ploidy_indices = np.asarray(focal["ploidy_index"])
        ancestor_ids = np.asarray(focal["ancestor_id"], dtype=str)
        offsets = np.asarray(focal["offsets"])
        ancestor_index = np.asarray(focal["ancestor_index"])
        derived_ac = np.asarray(focal["derived_ac"])

    num_rows = len(sample_ids)
    if sample_ids.ndim != 1 or ploidy_indices.ndim != 1:
        raise ValueError("Focal NPZ sample arrays must be one-dimensional; rebuild it")
    if len(ploidy_indices) != num_rows:
        raise ValueError(
            "Focal NPZ sample_id and ploidy_index are misaligned; rebuild it"
        )
    if not np.issubdtype(ploidy_indices.dtype, np.integer):
        raise ValueError("Focal NPZ ploidy_index must be integer; rebuild it")
    if not np.issubdtype(offsets.dtype, np.integer) or len(offsets) != num_rows + 1:
        raise ValueError("Focal NPZ offsets are invalid; rebuild it")
    if offsets.ndim != 1 or ancestor_index.ndim != 1:
        raise ValueError("Focal NPZ ragged arrays must be one-dimensional; rebuild it")
    if offsets[0] != 0 or offsets[-1] != len(ancestor_index):
        raise ValueError("Focal NPZ offsets do not bound ancestor_index; rebuild it")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("Focal NPZ offsets are not ordered; rebuild it")
    if not np.issubdtype(ancestor_index.dtype, np.integer):
        raise ValueError("Focal NPZ ancestor_index must be integer; rebuild it")
    if np.any(ancestor_index < 0) or np.any(ancestor_index >= len(ancestor_ids)):
        raise ValueError("Focal NPZ contains an invalid ancestor index; rebuild it")
    for row in range(num_rows):
        candidates = ancestor_index[offsets[row] : offsets[row + 1]]
        if len(candidates) > 1 and np.any(np.diff(candidates) <= 0):
            raise ValueError(
                f"Focal NPZ candidates for row {row} are not sorted distinct; rebuild it"
            )

    row_by_key = {}
    for row, (sample_id, ploidy_index) in enumerate(
        zip(sample_ids, ploidy_indices, strict=True)
    ):
        key = ("samples", str(sample_id), int(ploidy_index))
        if key in row_by_key:
            raise ValueError(
                f"Focal NPZ contains duplicate sample key {key}; rebuild it"
            )
        row_by_key[key] = row

    reference_ts = tskit.load(ancestors_ts_path)
    cfg = tsinfer.config.Config.from_toml(config_path)
    sample_match_config = cfg.match.sources["samples"]
    context = _build_context(
        reference_ts, ancestor_ids, derived_ac, ac_cutoff, sample_match_config
    )
    logger.info(
        "Evaluating %d haplotypes, %d ancestors, %d sites, and %d cutoffs with "
        "n=%d, mu=%g, rho=%g",
        num_rows,
        len(ancestor_ids),
        len(context.sites_position),
        len(context.cutoffs),
        context.num_reference_nodes,
        sample_match_config.mismatch,
        sample_match_config.recombination,
    )

    results = [None] * num_rows
    groups = set()
    with match_file_path.open() as match_file:
        for line_number, line in enumerate(match_file, start=1):
            record = json.loads(line)
            key = (
                record.get("source"),
                str(record.get("sample_id")),
                record.get("ploidy_index"),
            )
            row = row_by_key.get(key)
            if row is None:
                raise ValueError(
                    f"Unknown match key {key} at {match_file_path}:{line_number}"
                )
            if results[row] is not None:
                raise ValueError(
                    f"Duplicate match key {key} at {match_file_path}:{line_number}"
                )
            groups.add(record["group"])
            if len(groups) > 1:
                raise ValueError(
                    "Multiple sample match groups are not supported by this evaluator"
                )
            candidates = ancestor_index[offsets[row] : offsets[row + 1]]
            results[row] = _compute_haplotype_stats(record, candidates, context)
    missing_rows = [row for row, result in enumerate(results) if result is None]
    if missing_rows:
        row = missing_rows[0]
        key = ("samples", str(sample_ids[row]), int(ploidy_indices[row]))
        raise ValueError(
            f"Match file is missing focal row {key}; rebuild sample matches"
        )

    records = []
    for row, result in enumerate(results):
        for cutoff_index, cutoff in enumerate(context.cutoffs):
            fraction_bp = np.nan
            if result.evaluated_bp > 0:
                fraction_bp = result.covered_bp[cutoff_index] / result.evaluated_bp
            fraction_sites = np.nan
            if result.evaluated_sites > 0:
                fraction_sites = (
                    result.covered_sites[cutoff_index] / result.evaluated_sites
                )
            records.append(
                {
                    "source": "samples",
                    "sample_id": str(sample_ids[row]),
                    "ploidy_index": int(ploidy_indices[row]),
                    "ac_cutoff": int(cutoff),
                    "evaluated_bp": result.evaluated_bp,
                    "covered_bp": result.covered_bp[cutoff_index],
                    "fraction_covered_bp": fraction_bp,
                    "evaluated_sites": result.evaluated_sites,
                    "covered_sites": result.covered_sites[cutoff_index],
                    "fraction_covered_sites": fraction_sites,
                    "num_switches": result.num_switches,
                    "num_mismatches": result.num_mismatches,
                    "path_log_likelihood": result.path_log_likelihood,
                    "num_focal_ancestors": result.num_focal_ancestors[cutoff_index],
                }
            )
    return pd.DataFrame.from_records(records)
