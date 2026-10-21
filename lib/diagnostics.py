"""Reconstruct exact focal masks, compare raw paths and package one experiment.

Matching never calls this module. The driver uses the native roster and packed
reference helpers, and workers run only exact kernels and table-based analysis.
"""

import concurrent.futures as cf
import csv
import dataclasses
import datetime
import importlib.metadata
import json
import logging
import multiprocessing as mp
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd
import tsinfer
import tskit
import tszip
import yaml

from . import benchmark, greedy, haplotypes, matching, utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Run:
    """One configured scientific comparison and its reconstructed mask."""

    dataset: str
    af_cutoff: float
    min_region_size_kb: float
    stitched: pathlib.Path
    mask: pathlib.Path
    min_focal_region_size: int | None = None


@dataclasses.dataclass
class Inventory:
    """Concrete workflow products derived solely from the effective config."""

    full: list[pathlib.Path]
    runs: list[Run]
    directory: pathlib.Path
    archive: pathlib.Path
    tables: list[pathlib.Path]
    sites: list[pathlib.Path]


def inventory(config):
    """Use the same normalised grid in the DAG, analysis and archive manifest."""
    root = pathlib.Path(config["data_dir"]).expanduser()
    directory = root / "diagnostics" / config["config_name"]
    full = []
    runs = []
    sites = []
    for dataset in config["datasets"]:
        name = dataset["name"]
        full.append(root / "matches" / f"{name}_inferred_samples_raw.trees")
        sites.append(directory / f"{name}_sites.npz")
        for af in utils.parameter_values(config["af_cutoff"]):
            for minimum in utils.parameter_values(config["min_region_size_kb"]):
                parameters = pathlib.Path(f"af_{af}") / f"min_region_kb_{minimum}"
                stitched = root / "matches" / parameters
                mask = root / "focal_masks" / parameters
                runs.append(
                    Run(
                        name,
                        af,
                        minimum,
                        stitched / f"{name}_inferred_greedy_samples_raw.trees",
                        mask / f"{name}_focal_mask.npz",
                    )
                )
    tables = [
        directory / f"{name}.csv"
        for name in (
            "haplotype_results",
            "region_lengths",
            "run_summary",
            "sample_populations",
        )
    ]
    archive = root / "diagnostics" / f"{config['config_name']}_diagnostics.tar.gz"
    return Inventory(full, runs, directory, archive, tables, sites)


def population_sources(config):
    """Validate diagnostic-only population settings and declare external inputs."""
    paths = []
    for dataset in config["datasets"]:
        population = dataset.get("population")
        if population is None:
            raise ValueError(
                f"{dataset['name']}: diagnostics requires population settings"
            )
        source = population.get("source")
        if source == "ts":
            if dataset.get("ts_path") is None:
                raise ValueError(
                    f"{dataset['name']}: population.source ts requires ts_path"
                )
        elif source == "csv":
            paths.append(pathlib.Path(population["path"]).expanduser())
        else:
            raise ValueError(f"{dataset['name']}: population.source must be ts or csv")
        if not population.get("field"):
            raise ValueError(f"{dataset['name']}: population.field is required")
        if dataset.get("ts_path") is not None:
            paths.append(pathlib.Path(dataset["ts_path"]).expanduser())
    return list(dict.fromkeys(paths))


def read_population_metadata(dataset, source, jobs):
    """Join the actual native haplotypes to original TS or explicit CSV labels."""
    settings = dataset.get("population")
    if settings is None:
        raise ValueError(f"{dataset['name']}: diagnostics requires population settings")
    records = []
    if settings["source"] == "ts":
        store = tsinfer.vcz.open_store(source.path)
        ploidy = store["call_genotype"].shape[2]
        truth = tskit.load(pathlib.Path(dataset["ts_path"]).expanduser())
        mapping = truth.map_to_vcf_model()
        nodes = dict(
            zip(mapping.individuals_name, mapping.individuals_nodes, strict=True)
        )
        if len(nodes) != len(mapping.individuals_name):
            raise ValueError("Original TS individual names are not unique")
        for job in jobs:
            if job.sample_id not in nodes:
                raise ValueError(f"Selected sample {job.sample_id} is absent from truth")
            sample_nodes = nodes[job.sample_id]
            if len(sample_nodes) != ploidy or np.any(sample_nodes < 0):
                raise ValueError(f"Truth ploidy differs for {job.sample_id}")
            node_id = int(sample_nodes[job.ploidy_index])
            node = truth.node(node_id)
            if node.population == tskit.NULL:
                raise ValueError(f"Truth node {node.id} has no population")
            label = truth.population(node.population).metadata.get(settings["field"])
            records.append(
                {
                    "truth_node_id": node.id,
                    "truth_individual_id": node.individual,
                    "population_id": node.population,
                    "population": label,
                }
            )
    elif settings["source"] == "csv":
        store = tsinfer.vcz.open_store(source.path)
        columns = tsinfer.vcz.resolve_samples_selection(store, source.samples)
        ids = np.asarray(store["sample_id"].oindex[columns].tolist(), dtype=str)
        lookup = store[settings["zarr_id_field"]]
        if lookup.shape != store["sample_id"].shape:
            raise ValueError("Population join identifier must have the sample dimension")
        lookup_ids = np.asarray(lookup.oindex[columns].tolist(), dtype=str)
        if np.any(lookup_ids == "") or len(set(lookup_ids)) != len(lookup_ids):
            raise ValueError(
                "Selected population join identifiers are empty or duplicated"
            )
        pedigree = pd.read_csv(
            pathlib.Path(settings["path"]).expanduser(), dtype=str, keep_default_na=False
        )
        csv_ids = pedigree[settings["csv_id_field"]]
        if (csv_ids == "").any() or csv_ids.duplicated().any():
            raise ValueError("CSV identifiers are empty or duplicated")
        labels = pedigree.set_index(settings["csv_id_field"])[settings["field"]]
        selected = labels.reindex(lookup_ids)
        if selected.isna().any():
            raise ValueError("Selected samples have missing CSV identifiers")
        by_sample = dict(zip(ids, selected.tolist(), strict=True))
        records = [{"population": by_sample[job.sample_id]} for job in jobs]
    else:
        raise ValueError("population.source must be ts or csv")
    rows = []
    identities = set()
    for job, record in zip(jobs, records, strict=True):
        key = (job.source, job.sample_id, job.ploidy_index)
        if key in identities:
            raise ValueError(f"Duplicate selected identity: {key}")
        identities.add(key)
        label = record["population"]
        if label is None or pd.isna(label) or str(label).strip() == "":
            raise ValueError(f"Missing population label for {key}")
        rows.append(
            {
                "dataset": dataset["name"],
                "source": job.source,
                "sample_id": job.sample_id,
                "ploidy_index": job.ploidy_index,
                **record,
            }
        )
    return rows


@dataclasses.dataclass
class FocalMask:
    """Ragged maximal alternating focal/gap intervals on the reference site axis."""

    source: np.ndarray
    sample_id: np.ndarray
    ploidy_index: np.ndarray
    offsets: np.ndarray
    left_site_index: np.ndarray
    right_site_index: np.ndarray
    is_gap: np.ndarray
    domain_start: np.ndarray
    domain_end: np.ndarray
    num_sites: int
    af_cutoff: float
    min_region_size_kb: float
    min_focal_region_size: int
    focal_choice: str

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }
        np.savez(path, **values)

    @classmethod
    def load(cls, path):
        """Read the portable numeric/string representation without object decoding."""
        with np.load(path, allow_pickle=False) as saved:
            return cls(
                **{field.name: saved[field.name] for field in dataclasses.fields(cls)}
            )


@dataclasses.dataclass
class MaskRuns:
    """One haplotype's owned intervals before concatenating a pair's results."""

    left: np.ndarray
    right: np.ndarray
    is_gap: np.ndarray
    start: int
    end: int


def mask_runs(state, order, minimum, start, end):
    """Union touching cover runs, ignoring parents, then partition the domain."""
    count = greedy.cover(state, order, minimum)
    components = []
    for left, right in zip(state.run_left[:count], state.run_right[:count], strict=True):
        if components and left == components[-1][1]:
            components[-1][1] = int(right)
        else:
            components.append([int(left), int(right)])
    lefts, rights, gaps = [], [], []
    cursor = start
    for left, right in components:
        if cursor < left:
            lefts.append(cursor)
            rights.append(left)
            gaps.append(True)
        lefts.append(left)
        rights.append(right)
        gaps.append(False)
        cursor = right
    if cursor < end:
        lefts.append(cursor)
        rights.append(end)
        gaps.append(True)
    return MaskRuns(
        np.asarray(lefts, dtype=np.int64),
        np.asarray(rights, dtype=np.int64),
        np.asarray(gaps, dtype=bool),
        start,
        end,
    )


@dataclasses.dataclass
class SiteAxis:
    """Observed endpoint extent, intersected with reference inference intervals."""

    positions: np.ndarray
    inference_intervals: np.ndarray
    analysis_intervals: np.ndarray
    interval_index: np.ndarray
    sequence_length: float

    @classmethod
    def from_reference(cls, reference):
        positions = reference.sites_position
        if len(positions) == 0:
            raise ValueError("Diagnostics requires reference sites")
        intervals = np.asarray(reference.metadata["sequence_intervals"], dtype=float)
        if (
            intervals.ndim != 2
            or intervals.shape[1] != 2
            or np.any(intervals[:, 0] >= intervals[:, 1])
            or np.any(intervals[1:, 0] < intervals[:-1, 1])
        ):
            raise ValueError("Invalid reference inference intervals")
        clipped = intervals.copy()
        clipped[:, 0] = np.maximum(clipped[:, 0], positions[0])
        clipped[:, 1] = np.minimum(clipped[:, 1], positions[-1])
        clipped = clipped[clipped[:, 0] <= clipped[:, 1]]
        index = np.searchsorted(intervals[:, 0], positions, side="right") - 1
        present = (index >= 0) & (positions < intervals[np.maximum(index, 0), 1])
        index[~present] = -1
        return cls(positions, intervals, clipped, index, reference.sequence_length)

    @property
    def span_bp(self):
        return float(
            np.sum(self.analysis_intervals[:, 1] - self.analysis_intervals[:, 0])
        )

    def physical_bounds(self, left, right):
        """Apply matching endpoints before clipping to the observed analysis extent."""
        lo = 0.0 if left == 0 else float(self.positions[left])
        hi = (
            self.sequence_length
            if right == len(self.positions)
            else float(self.positions[right])
        )
        return np.asarray([lo, hi])

    def save(self, path):
        np.savez(
            path,
            positions=self.positions,
            inference_intervals=self.inference_intervals,
            analysis_intervals=self.analysis_intervals,
            sequence_length=self.sequence_length,
            analysis_start=self.positions[0],
            analysis_end=self.positions[-1],
        )


@dataclasses.dataclass
class EdgePath:
    """Sorted child edges as views of one grouped numerical edge array."""

    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray

    def parents_at_sites(self, positions):
        result = np.full(len(positions), tskit.NULL, dtype=np.int32)
        if len(self.left) > 0:
            index = np.searchsorted(self.left, positions, side="right") - 1
            valid = (index >= 0) & (positions < self.right[np.maximum(index, 0)])
            result[valid] = self.parent[index[valid]]
        return result

    def support(self):
        """Discard parent changes and join touching intervals in linear time."""
        intervals = []
        for left, right in zip(self.left, self.right, strict=True):
            if intervals and left == intervals[-1][1]:
                intervals[-1][1] = float(right)
            else:
                intervals.append([float(left), float(right)])
        return np.asarray(intervals, dtype=float).reshape(-1, 2)


def read_paths(path, reference, identities):
    """Index sample edges once using tskit's immutable, zero-copy columns.

    See https://tskit.dev/tskit/docs/stable/python-api.html#efficient-table-column-access
    and the half-open edge requirements in the tskit data model. Raw nodes are
    validated against the reference; no simplification or tree traversal is needed.
    """
    ts = tskit.load(path)
    if not np.array_equal(ts.sites_position, reference.sites_position):
        raise ValueError(f"Matched/reference site axes disagree: {path}")
    if ts.tables.nodes[: reference.num_nodes] != reference.tables.nodes:
        raise ValueError(f"Matched tree changed reference nodes: {path}")
    nodes = {}
    for node_id in ts.samples():
        node = ts.node(int(node_id))
        metadata = node.metadata
        key = (
            metadata.get("source"),
            metadata.get("sample_id"),
            metadata.get("ploidy_index"),
        )
        if key in nodes:
            raise ValueError(f"Duplicate matched identity {key}: {path}")
        nodes[key] = node.id
    if nodes.keys() != set(identities):
        raise ValueError(
            f"Matched sample roster disagrees with native selection: {path}"
        )
    selected = np.zeros(ts.num_nodes, dtype=bool)
    selected[list(nodes.values())] = True
    edges = np.flatnonzero(selected[ts.edges_child])
    order = np.lexsort((ts.edges_left[edges], ts.edges_child[edges]))
    edges = edges[order]
    children = ts.edges_child[edges]
    left, right, parents = (
        ts.edges_left[edges],
        ts.edges_right[edges],
        ts.edges_parent[edges],
    )
    if np.any(parents >= reference.num_nodes):
        raise ValueError(f"Sample path contains a non-reference parent: {path}")
    result = {}
    for key, node in nodes.items():
        first, last = np.searchsorted(children, [node, node + 1])
        path = EdgePath(left[first:last], right[first:last], parents[first:last])
        if np.any(path.left[1:] < path.right[:-1]):
            raise ValueError(f"Overlapping sample edges for {key}")
        result[key] = path
    return result


@dataclasses.dataclass
class IndexedPaths:
    """Read-only ragged copying paths in canonical haplotype order."""

    offsets: np.ndarray
    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray

    def __getitem__(self, index):
        first, last = self.offsets[index : index + 2]
        return EdgePath(
            self.left[first:last], self.right[first:last], self.parent[first:last]
        )


@dataclasses.dataclass
class PathIndex:
    """Location of shared columns prepared by :func:`prepare_paths`."""

    directory: pathlib.Path

    def open(self):
        arrays = [
            np.load(self.directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("offsets", "left", "right", "parent")
        ]
        return IndexedPaths(*arrays)


def prepare_paths(path, reference, identities, directory):
    """Validate once and copy paths directly into canonical, shared edge columns."""
    started = time.perf_counter()
    paths = read_paths(path, reference, identities)
    directory.mkdir()
    offsets = np.zeros(len(identities) + 1, dtype=np.int64)
    lengths = [len(paths[key].left) for key in identities]
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    np.save(directory / "offsets.npy", offsets, allow_pickle=False)
    for name, dtype in (
        ("left", np.float64),
        ("right", np.float64),
        ("parent", np.int32),
    ):
        column = np.lib.format.open_memmap(
            directory / f"{name}.npy", mode="w+", dtype=dtype, shape=(int(offsets[-1]),)
        )
        try:
            for index, key in enumerate(identities):
                first, last = offsets[index : index + 2]
                column[first:last] = getattr(paths[key], name)
        finally:
            haplotypes.close_mapping(column)
    logger.info(
        "Prepared diagnostic paths %s: %d haplotypes, %d edges in %.3fs",
        path,
        len(identities),
        offsets[-1],
        time.perf_counter() - started,
    )
    return PathIndex(directory)


def intersect_intervals(first, second):
    """Intersect two sorted disjoint physical interval lists in linear time."""
    result = []
    i, j = 0, 0
    while i < len(first) and j < len(second):
        left = max(first[i][0], second[j][0])
        right = min(first[i][1], second[j][1])
        if left < right:
            result.append([left, right])
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return np.asarray(result, dtype=float).reshape(-1, 2)


def ratio(numerator, denominator, scale=1):
    return scale * numerator / denominator if denominator > 0 else np.nan


def region_rows(mask, axis, common, valid):
    """Split mask regions at excluded/unmatched spans, retaining endpoint sites."""
    rows = []
    # Each list is ordered; search by interval right ends to avoid a quadratic scan.
    for region_index, (left, right, gap) in enumerate(
        zip(mask.left, mask.right, mask.is_gap, strict=True)
    ):
        bounds = axis.physical_bounds(left, right)
        first = np.searchsorted(common[:, 1], bounds[0], side="right")
        last = np.searchsorted(common[:, 0], bounds[1], side="left")
        for support in common[first:last]:
            lo = max(bounds[0], support[0])
            hi = min(bounds[1], support[1])
            first_interval = np.searchsorted(
                axis.inference_intervals[:, 1], lo, side="right"
            )
            last_interval = np.searchsorted(
                axis.inference_intervals[:, 0], hi, side="left"
            )
            for extent in axis.inference_intervals[first_interval:last_interval]:
                start = max(lo, extent[0], axis.positions[0])
                physical_end = min(hi, extent[1])
                end = min(physical_end, axis.positions[-1])
                if start > end:
                    continue
                site_start = max(int(left), int(np.searchsorted(axis.positions, start)))
                # Count the last observed site only when the original half-open
                # ancestry and inference interval include it, before BP clipping.
                site_end = min(
                    int(right), int(np.searchsorted(axis.positions, physical_end))
                )
                count = int(np.count_nonzero(valid[site_start:site_end]))
                if start == end and count == 0:
                    continue
                rows.append(
                    {
                        "region_index": region_index,
                        "is_gap": bool(gap),
                        "left_site_index": site_start,
                        "right_site_index": site_end,
                        "num_sites": count,
                        "left_bp": float(start),
                        "right_bp": float(end),
                        "span_bp": float(end - start),
                    }
                )
    return rows


def comparison_metrics(full, stitched, full_support, stitched_path, mask, axis):
    """Compare adjacent valid parents, with a shared mask and physical domain."""
    domain = np.zeros(len(full), dtype=bool)
    domain[mask.start : mask.end] = True
    inferred = axis.interval_index >= 0
    valid = domain & inferred & (full >= 0) & (stitched >= 0)
    focal = np.zeros(len(full), dtype=bool)
    for left, right in zip(
        mask.left[~mask.is_gap], mask.right[~mask.is_gap], strict=True
    ):
        focal[left:right] = True
    common = intersect_intervals(full_support, stitched_path.support())
    continuity = np.zeros(len(full) - 1, dtype=bool)
    if len(common) > 0:
        index = np.searchsorted(common[:, 0], axis.positions, side="right") - 1
        continuity = (index[1:] == index[:-1]) & (index[:-1] >= 0)
    adjacent = valid[1:] & valid[:-1] & continuity
    adjacent &= axis.interval_index[1:] == axis.interval_index[:-1]
    metrics = {}
    for method, parents in (("full", full), ("stitched", stitched)):
        changes = (parents[1:] != parents[:-1]) & adjacent
        metrics[f"{method}_switches"] = int(np.count_nonzero(changes))
        metrics[f"{method}_focal_switches"] = int(
            np.count_nonzero(changes & focal[1:] & focal[:-1])
        )
        metrics[f"{method}_gap_switches"] = int(
            np.count_nonzero(changes & ~focal[1:] & ~focal[:-1])
        )
        metrics[f"{method}_seam_switches"] = int(
            np.count_nonzero(changes & (focal[1:] != focal[:-1]))
        )
    extra = metrics["stitched_switches"] - metrics["full_switches"]
    metrics.update(
        num_extra_switches=extra,
        switch_increase_percentage=ratio(extra, metrics["full_switches"], 100),
        extra_switch_fraction=ratio(extra, metrics["stitched_switches"]),
        domain_start=mask.start,
        domain_end=mask.end,
        domain_sites=mask.end - mask.start,
        num_sites=len(full),
        inference_sites=int(np.count_nonzero(inferred)),
        excluded_sites=int(np.count_nonzero(~inferred)),
        evaluated_sites=int(np.count_nonzero(valid)),
        unmatched_sites=int(np.count_nonzero(inferred & ~valid)),
        outside_domain_sites=int(np.count_nonzero(inferred & ~domain)),
        domain_unmatched_sites=int(np.count_nonzero(inferred & domain & ~valid)),
        full_unmatched_sites=int(np.count_nonzero(inferred & domain & (full < 0))),
        stitched_unmatched_sites=int(
            np.count_nonzero(inferred & domain & (stitched < 0))
        ),
        focal_mask_sites=int(np.count_nonzero(focal & domain)),
        gap_mask_sites=int(np.count_nonzero(~focal & domain)),
    )
    disagreements = (full != stitched) & valid
    metrics["disagreement_sites"] = int(np.count_nonzero(disagreements))
    metrics["parents_identical"] = (
        metrics["disagreement_sites"] == 0 if metrics["evaluated_sites"] > 0 else None
    )
    metrics["parent_agreement_percentage"] = ratio(
        metrics["evaluated_sites"] - metrics["disagreement_sites"],
        metrics["evaluated_sites"],
        100,
    )
    for name, selected in (("focal", focal), ("gap", ~focal)):
        count = int(np.count_nonzero(valid & selected))
        disagree = int(np.count_nonzero(disagreements & selected))
        metrics[f"{name}_sites"] = count
        metrics[f"{name}_disagreement_sites"] = disagree
        metrics[f"{name}_agreement_percentage"] = ratio(count - disagree, count, 100)
    # Restrict physical supports to the matched domain before clipping region rows.
    bounds = axis.physical_bounds(mask.start, mask.end).reshape(1, 2)
    common = intersect_intervals(common, bounds)
    lengths = region_rows(mask, axis, common, valid)
    for name, gap in (("focal", False), ("gap", True)):
        span = sum(row["span_bp"] for row in lengths if row["is_gap"] == gap)
        metrics[f"{name}_span_bp"] = span
        metrics[f"{name}_coverage_percentage"] = ratio(span, axis.span_bp, 100)
    matched = metrics["focal_span_bp"] + metrics["gap_span_bp"]
    metrics["analysis_start_bp"] = float(axis.positions[0])
    metrics["analysis_end_bp"] = float(axis.positions[-1])
    metrics["analysis_span_bp"] = axis.span_bp
    domain_intervals = intersect_intervals(bounds, axis.analysis_intervals)
    domain_span = float(np.sum(domain_intervals[:, 1] - domain_intervals[:, 0]))
    metrics["domain_span_bp"] = domain_span
    metrics["evaluated_span_bp"] = matched
    metrics["outside_domain_span_bp"] = max(0.0, axis.span_bp - domain_span)
    metrics["domain_unmatched_span_bp"] = max(0.0, domain_span - matched)
    metrics["excluded_span_bp"] = float(
        axis.positions[-1] - axis.positions[0] - axis.span_bp
    )
    metrics["unmatched_span_bp"] = max(0.0, axis.span_bp - matched)
    metrics["unmatched_coverage_percentage"] = ratio(
        metrics["unmatched_span_bp"], axis.span_bp, 100
    )
    return HaplotypeAnalysis(metrics, lengths)


@dataclasses.dataclass
class HaplotypeAnalysis:
    metrics: dict
    regions: list[dict]


@dataclasses.dataclass
class WorkerData:
    """Private exact scratch and read-only shared inputs persist across tasks."""

    samples: haplotypes.SampleInput
    directory: pathlib.Path
    panel_path: pathlib.Path
    exact: greedy.ExactState
    axis: SiteAxis
    labels: list[dict]
    runs: list[Run]
    full: IndexedPaths
    stitched: dict[float, IndexedPaths]


@dataclasses.dataclass
class HaplotypeDiagnostic:
    index: int
    min_region_size_kb: float
    mask: MaskRuns
    row: dict
    regions: list[dict]


@dataclasses.dataclass
class HaplotypeTask:
    """One diagnostic job referencing a complete row in a shared sample slab."""

    slab: haplotypes.MappedCalls
    row: int
    job_index: int


_worker: WorkerData


def _serial_helpers(helpers):
    utils.configure_helpers({name: 1 for name in helpers})


def _initialise_worker(
    samples,
    directory,
    panel_path,
    axis,
    full_index,
    stitched_indexes,
    catalog,
    labels,
    runs,
    helpers,
):
    global _worker
    _serial_helpers(helpers)
    _worker = WorkerData(
        samples,
        directory,
        panel_path,
        greedy.make_state(catalog),
        axis,
        labels,
        runs,
        full_index.open(),
        {minimum: index.open() for minimum, index in stitched_indexes.items()},
    )


def _pack_chunk(chunk):
    return haplotypes.pack_reference_chunk(
        _worker.panel_path, _worker.exact.catalog, chunk
    )


def _decode_chunk(chunk):
    return haplotypes.load_sample_chunk(_worker.samples, chunk, _worker.directory)


def _analyse_haplotype(task):
    """Extend, sort and prepare baseline support once for all this row's minima."""
    state = _worker
    calls = task.slab.open()
    haplotype = calls[task.row]
    results = []
    try:
        index = task.job_index
        called = np.flatnonzero(haplotype >= 0)
        start, end = 0, len(haplotype)
        if len(called) > 0:
            start, end = int(called[0]), int(called[-1]) + 1
        order = greedy.candidates(haplotype, state.exact)
        label = state.labels[index]
        full_path = state.full[index]
        full = full_path.parents_at_sites(state.axis.positions)
        full_support = full_path.support()
        for run in state.runs:
            minimum = run.min_focal_region_size
            mask = mask_runs(state.exact, order, minimum, start, end)
            stitched_path = state.stitched[run.min_region_size_kb][index]
            stitched = stitched_path.parents_at_sites(state.axis.positions)
            analysis = comparison_metrics(
                full, stitched, full_support, stitched_path, mask, state.axis
            )
            identity = {
                "dataset": run.dataset,
                "af_cutoff": run.af_cutoff,
                "min_region_size_kb": run.min_region_size_kb,
                "min_focal_region_size": minimum,
                "source": label["source"],
                "sample_id": label["sample_id"],
                "ploidy_index": label["ploidy_index"],
                "population": label["population"],
            }
            results.append(
                HaplotypeDiagnostic(
                    index,
                    run.min_region_size_kb,
                    mask,
                    {**identity, **analysis.metrics},
                    [{**identity, **region} for region in analysis.regions],
                )
            )
    finally:
        del haplotype
        haplotypes.close_mapping(calls)
    return results


def _sample_waves(chunks, workers):
    """Admit complete physical chunks until at least one row per worker is ready."""
    wave = []
    count = 0
    for chunk in chunks:
        wave.append(chunk)
        count += len(chunk.job_indices)
        if count >= workers:
            yield wave
            wave = []
            count = 0
    if wave:
        yield wave


def _save_mask(path, results, jobs, num_sites, run, focal_choice):
    masks = [result.mask for result in results]
    offsets = np.zeros(len(masks) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(mask.left) for mask in masks])
    mask = FocalMask(
        np.asarray([job.source for job in jobs], dtype=str),
        np.asarray([job.sample_id for job in jobs], dtype=str),
        np.asarray([job.ploidy_index for job in jobs], dtype=np.int32),
        offsets,
        np.concatenate([mask.left for mask in masks]),
        np.concatenate([mask.right for mask in masks]),
        np.concatenate([mask.is_gap for mask in masks]),
        np.asarray([mask.start for mask in masks], dtype=np.int64),
        np.asarray([mask.end for mask in masks], dtype=np.int64),
        num_sites,
        run.af_cutoff,
        run.min_region_size_kb,
        run.min_focal_region_size,
        focal_choice,
    )
    mask.save(path)


def _summary(rows):
    """Equal haplotype weights; SEM describes variability, not independence."""
    frame = pd.DataFrame(rows)
    first = rows[0]
    result = {
        key: first[key]
        for key in (
            "dataset",
            "af_cutoff",
            "min_region_size_kb",
            "min_focal_region_size",
        )
    }
    result["num_haplotypes"] = len(rows)
    result["num_samples"] = len(frame[["source", "sample_id"]].drop_duplicates())
    for name in ("focal", "gap"):
        result[f"empty_{name}_haplotypes"] = int((frame[f"{name}_sites"] == 0).sum())
    result["empty_evaluated_haplotypes"] = int((frame["evaluated_sites"] == 0).sum())
    result["zero_full_switch_haplotypes"] = int((frame["full_switches"] == 0).sum())
    identity_fields = {
        "dataset",
        "af_cutoff",
        "min_region_size_kb",
        "min_focal_region_size",
        "source",
        "sample_id",
        "ploidy_index",
        "population",
        "parents_identical",
    }
    for name in frame.columns:
        if name in identity_fields:
            continue
        values = frame[name]
        result[f"{name}_mean"] = values.mean()
        result[f"{name}_sem"] = values.sem()
        result[f"{name}_count"] = int(values.count())
        if name.endswith(("_percentage", "_fraction")):
            result[f"{name}_undefined_count"] = int(values.isna().sum())
    result["parents_identical_count"] = int(
        frame["parents_identical"].fillna(False).sum()
    )
    result["parents_identical_undefined_count"] = int(
        frame["parents_identical"].isna().sum()
    )
    return result


def _write_rows(path, rows, fields, append=False):
    with path.open("a" if append else "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


@dataclasses.dataclass
class ArchiveFile:
    """One unique source and its portable destination, before tree compression."""

    source: pathlib.Path
    archive: str
    tree: bool


def _versions():
    versions = {}
    for name in (
        "tsinfer",
        "tskit",
        "tszip",
        "numpy",
        "numba",
        "zarr",
        "numcodecs",
        "pandas",
        "snakemake",
        "pyyaml",
    ):
        versions[name] = importlib.metadata.version(name)
    revisions = {}
    roots = {
        "pipeline": pathlib.Path(__file__).resolve().parent.parent,
        "tsinfer": pathlib.Path(tsinfer.__file__).resolve().parent.parent,
    }
    for name, root in roots.items():
        if (root / ".git").exists():
            revisions[name] = benchmark._revision(root)
    return {"python": sys.version, "dependencies": versions, "revisions": revisions}


def _performance_files(config):
    """Only this configured, completed suite; prune every Zarr store directory."""
    if config.get("performance") is None:
        return []
    for path in benchmark.final_products(config):
        if not path.is_file():
            raise ValueError(
                f"Configured performance suite is incomplete: missing {path}"
            )
    root = benchmark.output_directory(config)
    runs = pd.read_csv(root / "runs.csv")
    complete = json.loads((root / "complete.json").read_text())
    if len(runs) != complete["runs"] or not (runs["status"] == "ok").all():
        raise ValueError("Configured performance suite has incomplete or failed trials")
    for run_id in runs["run_id"]:
        tree = root / run_id / "samples_raw.trees"
        if not tree.is_file():
            raise ValueError(
                f"Configured performance suite is missing trial tree: {tree}"
            )
    paths = []
    for directory, directories, filenames in os.walk(root):
        parent = pathlib.Path(directory)
        directories[:] = [
            name
            for name in directories
            if not (
                name.endswith(".zarr")
                or (parent / name / "zarr.json").exists()
                or (parent / name / ".zgroup").exists()
                or (parent / name / ".zarray").exists()
            )
        ]
        paths.extend(parent / name for name in filenames if not name.endswith(".bin"))
    return sorted(paths)


def _archive_manifest(config, products, datasets, workers):
    """Deduplicate paths, retaining distinct benchmark trial identities."""
    files = {}
    destinations = set()

    def add(source, destination, tree=False):
        source = source.expanduser().resolve()
        if source in files:
            return files[source].archive
        if not source.is_file():
            raise FileNotFoundError(source)
        if tree:
            destination += ".tsz"
        if destination in destinations:
            raise ValueError(f"Archive destination collision: {destination}")
        destinations.add(destination)
        files[source] = ArchiveFile(source, destination, tree)
        return destination

    snapshot = products.directory / "config.yaml"
    manifest_path = products.directory / "manifest.json"
    add(snapshot, "config.yaml")
    # The manifest itself is added after it is written, below.
    table_map = {
        path.stem: add(path, f"diagnostics/{path.name}") for path in products.tables
    }
    dataset_map = []
    for dataset, information in zip(config["datasets"], datasets, strict=True):
        name = dataset["name"]
        inputs = benchmark.prepared_inputs(config, dataset)
        full_path = (
            pathlib.Path(config["data_dir"]).expanduser()
            / "matches"
            / f"{name}_inferred_samples_raw.trees"
        )
        entry = {
            "name": name,
            "native_config": add(inputs.config, f"configs/{inputs.config.name}"),
            "reference": add(
                inputs.reference, f"ancestors/{inputs.reference.name}", True
            ),
            "full_hmm": add(full_path, f"matches/{full_path.name}", True),
            "sites": add(
                products.directory / f"{name}_sites.npz", f"diagnostics/{name}_sites.npz"
            ),
            "population_provenance": dataset["population"],
            **information,
            "runs": [],
        }
        truth = dataset.get("ts_path")
        entry["truth"] = None
        if truth is not None:
            entry["truth"] = add(pathlib.Path(truth), f"truth/{name}.trees", True)
        for run in products.runs:
            if run.dataset == name:
                relative = run.stitched.relative_to(
                    pathlib.Path(config["data_dir"]).expanduser()
                )
                relative_mask = run.mask.relative_to(
                    pathlib.Path(config["data_dir"]).expanduser()
                )
                entry["runs"].append(
                    {
                        "af_cutoff": run.af_cutoff,
                        "min_region_size_kb": run.min_region_size_kb,
                        "min_focal_region_size": run.min_focal_region_size,
                        "stitched": add(run.stitched, relative.as_posix(), True),
                        "mask": add(run.mask, relative_mask.as_posix()),
                    }
                )
        dataset_map.append(entry)
    performance_files = []
    for path in _performance_files(config):
        relative = path.relative_to(benchmark.output_directory(config))
        destination = (
            pathlib.Path("performance")
            / config["performance"]["experiment_id"]
            / relative
        )
        performance_files.append(
            add(path, destination.as_posix(), path.suffix == ".trees")
        )
    manifest = {
        "config_name": config["config_name"],
        "config": "config.yaml",
        "diagnostics_time_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "diagnostics_time_versions": _versions(),
        "diagnostics_time_execution": {
            "allocated_workers": workers,
            "helper_threads": {name: 1 for name in config["helper_threads"]},
        },
        "version_scope": (
            "Captured during diagnostics; benchmark manifests retain original versions. "
            "These versions do not establish which code produced older trees."
        ),
        "datasets": dataset_map,
        "tables": table_map,
        "performance": performance_files,
        "sem_scope": (
            "Haplotype variability with equal weights; within-individual dependence "
            "is not accounted for."
        ),
        "files": [],
    }
    # Include the self mapping without depending on a previously existing manifest.
    files[manifest_path.resolve()] = ArchiveFile(
        manifest_path.resolve(), "manifest.json", False
    )
    manifest["files"] = [
        {"source": str(item.source), "archive": item.archive, "tree": item.tree}
        for item in files.values()
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return list(files.values())


def _compress_tree(task):
    source, destination = task
    destination.parent.mkdir(parents=True, exist_ok=True)
    ts = tskit.load(source)
    tszip.compress(ts, destination)


def package_archive(files, archive, workers, helpers):
    """Stage unique tszipped trees, then stream tar through level-1 pigz."""
    started = time.perf_counter()
    pigz = shutil.which("pigz")
    if pigz is None:
        raise RuntimeError(
            "Diagnostics requires pigz on PATH (compression uses level 1)"
        )
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp")
    try:
        with tempfile.TemporaryDirectory(prefix="diagnostics_tsz_") as staging:
            staging = pathlib.Path(staging)
            trees = [
                (item.source, staging / item.archive) for item in files if item.tree
            ]
            logger.info(
                "Compressing %d unique trees with tszip using up to %d workers",
                len(trees),
                workers,
            )
            with cf.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=_serial_helpers,
                initargs=(helpers,),
            ) as pool:
                futures = {
                    pool.submit(_compress_tree, task): task[1].relative_to(staging)
                    for task in trees
                }
                completed = 0
                for future in cf.as_completed(futures):
                    future.result()
                    completed += 1
                    logger.info(
                        "Tree compression %d/%d complete: %s",
                        completed,
                        len(trees),
                        futures[future],
                    )
            logger.info(
                "Tree compression complete; streaming %d files through pigz "
                "(%d threads, level 1) into %s",
                len(files),
                workers,
                archive,
            )
            with temporary.open("wb") as output, tempfile.TemporaryFile() as errors:
                with subprocess.Popen(
                    [pigz, "-p", str(workers), "-1"],
                    stdin=subprocess.PIPE,
                    stdout=output,
                    stderr=errors,
                ) as compressor:
                    pipe_error = None
                    try:
                        with tarfile.open(fileobj=compressor.stdin, mode="w|") as tar:
                            for index, item in enumerate(files, start=1):
                                source = (
                                    staging / item.archive if item.tree else item.source
                                )
                                logger.info(
                                    "Archiving file %d/%d: %s (%d bytes)",
                                    index,
                                    len(files),
                                    item.archive,
                                    source.stat().st_size,
                                )
                                tar.add(source, arcname=item.archive, recursive=False)
                    except BrokenPipeError as error:
                        pipe_error = error
                    finally:
                        try:
                            compressor.stdin.close()
                        except BrokenPipeError as error:
                            pipe_error = error
                    logger.info("Tar stream written; waiting for pigz to finish")
                    status = compressor.wait()
                    if status != 0:
                        errors.seek(0)
                        message = errors.read().decode()
                        raise RuntimeError(f"pigz exited {status}: {message}")
                    if pipe_error is not None:
                        raise pipe_error
            logger.info("Tar/pigz finished; removing staged tszip files")
        temporary.replace(archive)
        logger.info(
            "Archive complete: %s (%d bytes), compression/packaging took %.1fs",
            archive,
            archive.stat().st_size,
            time.perf_counter() - started,
        )
    finally:
        temporary.unlink(missing_ok=True)


def run(config, workers):
    """Produce the configured grid and archive after all workflow inputs exist."""
    utils.validate_config(config)
    if shutil.which("pigz") is None:
        raise RuntimeError("Diagnostics requires pigz on PATH")
    population_sources(config)
    _performance_files(config)
    products = inventory(config)
    products.directory.mkdir(parents=True, exist_ok=True)
    (products.directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    _serial_helpers(config["helper_threads"])
    temp_dir = config.get("temp_dir")
    if temp_dir is not None:
        temp_dir = pathlib.Path(temp_dir).expanduser()
        temp_dir.mkdir(parents=True, exist_ok=True)
    first_pair = True
    population_rows = []
    manifest_datasets = []
    for dataset, full_path in zip(config["datasets"], products.full, strict=True):
        inputs = benchmark.prepared_inputs(config, dataset)
        native = tsinfer.config.Config.from_toml(inputs.config)
        cfg = matching._sample_config(inputs.config, inputs.reference, full_path)
        reference = tskit.load(inputs.reference)
        axis = SiteAxis.from_reference(reference)
        for item in products.runs:
            if item.dataset == dataset["name"]:
                item.min_focal_region_size = utils.min_region_size_sites(
                    axis.positions, item.min_region_size_kb
                )
        jobs, _, _ = tsinfer.pipeline._build_jobs_with_metadata(
            cfg, axis.positions.astype(np.int32)
        )
        jobs.sort(key=lambda job: job.haplotype_index)
        if len(jobs) == 0:
            raise ValueError("Diagnostics requires selected haplotypes")
        samples = haplotypes.prepare_samples(cfg.sources["samples"], reference, jobs)
        labels = read_population_metadata(dataset, cfg.sources["samples"], jobs)
        population_rows.extend(labels)
        axis.save(products.directory / f"{dataset['name']}_sites.npz")
        manifest_datasets.append(
            {
                "cohort": labels,
                "num_sites": len(axis.positions),
                "reference_site_axis": "positions in sites NPZ",
                "analysis_extent_bp": [
                    float(axis.positions[0]),
                    float(axis.positions[-1]),
                ],
                "analysis_intervals": axis.analysis_intervals.tolist(),
                "analysis_span_bp": axis.span_bp,
                "reference_inference_intervals": axis.inference_intervals.tolist(),
            }
        )
        identities = [(job.source, job.sample_id, job.ploidy_index) for job in jobs]
        with tempfile.TemporaryDirectory(
            dir=temp_dir, prefix="diagnostics_dataset_"
        ) as dataset_temporary:
            dataset_directory = pathlib.Path(dataset_temporary)
            full_index = prepare_paths(
                full_path, reference, identities, dataset_directory / "full"
            )
            for af in utils.parameter_values(config["af_cutoff"]):
                runs = [
                    item
                    for item in products.runs
                    if item.dataset == dataset["name"] and item.af_cutoff == af
                ]
                logger.info(
                    "Diagnostics %s AF=%s: %d haplotypes, %d physical sample chunks, "
                    "%d workers, %d analysis tasks",
                    dataset["name"],
                    af,
                    len(jobs),
                    len(samples.chunks),
                    workers,
                    len(jobs),
                )
                with tempfile.TemporaryDirectory(
                    dir=dataset_directory, prefix="diagnostics_af_"
                ) as temporary:
                    directory = pathlib.Path(temporary)
                    stitched_indexes = {
                        item.min_region_size_kb: prepare_paths(
                            item.stitched,
                            reference,
                            identities,
                            directory / f"stitched_kb_{item.min_region_size_kb}",
                        )
                        for item in runs
                    }
                    packing_started = time.perf_counter()
                    catalog = haplotypes.prepare_reference(
                        native.ancestors[0].path,
                        samples,
                        reference,
                        af,
                        config.get("focal_choice", "left"),
                        directory,
                    )
                    results = {
                        item.min_region_size_kb: [None] * len(jobs) for item in runs
                    }
                    decoding_s = 0.0
                    analysis_s = 0.0
                    blocks_read = 0
                    with cf.ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=mp.get_context("spawn"),
                        initializer=_initialise_worker,
                        initargs=(
                            samples,
                            directory,
                            native.ancestors[0].path,
                            axis,
                            full_index,
                            stitched_indexes,
                            catalog,
                            labels,
                            runs,
                            config["helper_threads"],
                        ),
                    ) as pool:
                        chunks = haplotypes.reference_chunks(
                            native.ancestors[0].path, catalog
                        )
                        for _ in pool.map(_pack_chunk, chunks):
                            pass
                        logger.info(
                            "Diagnostic reference preparation/packing "
                            "(including worker startup): %.3fs",
                            time.perf_counter() - packing_started,
                        )
                        for wave in _sample_waves(samples.chunks, workers):
                            started = time.perf_counter()
                            slabs = list(pool.map(_decode_chunk, wave))
                            decoding_s += time.perf_counter() - started
                            blocks_read += sum(slab.blocks_read for slab in slabs)
                            tasks = [
                                HaplotypeTask(slab.calls, row, int(index))
                                for slab in slabs
                                for row, index in enumerate(slab.job_indices)
                            ]
                            started = time.perf_counter()
                            for haplotype_results in pool.map(
                                _analyse_haplotype, tasks, chunksize=1
                            ):
                                for result in haplotype_results:
                                    minimum_results = results[result.min_region_size_kb]
                                    minimum_results[result.index] = result
                            analysis_s += time.perf_counter() - started
                            for slab in slabs:
                                slab.calls.path.unlink()
                    logger.info(
                        "Diagnostic sample decoding: %.3fs, %d genotype tiles; "
                        "haplotype analysis: %.3fs",
                        decoding_s,
                        blocks_read,
                        analysis_s,
                    )
                    writing_started = time.perf_counter()
                    rows, lengths, summaries = [], [], []
                    for item in runs:
                        selected = results[item.min_region_size_kb]
                        if any(result is None for result in selected):
                            raise RuntimeError(
                                "Diagnostics did not return every selected haplotype"
                            )
                        _save_mask(
                            item.mask,
                            selected,
                            jobs,
                            len(axis.positions),
                            item,
                            config.get("focal_choice", "left"),
                        )
                        run_rows = [result.row for result in selected]
                        rows.extend(run_rows)
                        lengths.extend(
                            region for result in selected for region in result.regions
                        )
                        summaries.append(_summary(run_rows))
                    region_fields = [
                        "dataset",
                        "af_cutoff",
                        "min_region_size_kb",
                        "min_focal_region_size",
                        "source",
                        "sample_id",
                        "ploidy_index",
                        "population",
                        "region_index",
                        "is_gap",
                        "left_site_index",
                        "right_site_index",
                        "num_sites",
                        "left_bp",
                        "right_bp",
                        "span_bp",
                    ]
                    _write_rows(products.tables[0], rows, list(rows[0]), not first_pair)
                    _write_rows(
                        products.tables[1], lengths, region_fields, not first_pair
                    )
                    _write_rows(
                        products.tables[2], summaries, list(summaries[0]), not first_pair
                    )
                    first_pair = False
                    del results, selected, rows, lengths, summaries
                    logger.info(
                        "Diagnostic mask/table writing: %.3fs",
                        time.perf_counter() - writing_started,
                    )
    population_fields = [
        "dataset",
        "source",
        "sample_id",
        "ploidy_index",
        "population",
        "truth_node_id",
        "truth_individual_id",
        "population_id",
    ]
    _write_rows(products.tables[3], population_rows, population_fields)
    packaging_started = time.perf_counter()
    files = _archive_manifest(config, products, manifest_datasets, workers)
    package_archive(files, products.archive, workers, config["helper_threads"])
    logger.info(
        "Diagnostic archive compression/packaging: %.3fs; wrote %s",
        time.perf_counter() - packaging_started,
        products.archive,
    )
    return products.archive
