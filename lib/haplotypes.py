"""Packed ancestor references and physical-block, selected-cohort sample loading."""

import concurrent.futures as cf
import dataclasses
import fcntl
import functools
import hashlib
import itertools
import logging
import os
import pathlib
import tempfile
import time

import numba
import numpy as np
import tsinfer

from . import performance

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MappedCalls:
    """A temporary int8 input; empty references use an ordinary empty array."""

    path: pathlib.Path
    shape: tuple[int, ...]

    def open(self, mode="r"):
        if np.prod(self.shape) == 0:
            calls = np.empty(self.shape, dtype=np.int8)
            calls.flags.writeable = mode != "r"
            return calls
        return np.memmap(self.path, dtype=np.int8, mode=mode, shape=self.shape)


def close_mapping(calls):
    """Close a mapping only after all row views have been released."""
    if isinstance(calls, np.memmap):
        calls._mmap.close()


@dataclasses.dataclass
class ReferenceCatalog:
    """Eligible ancestors in original panel-column order; bounds are site indices."""

    parents: np.ndarray
    columns: np.ndarray
    seeds: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    offsets: np.ndarray
    packed: MappedCalls

    def read_packed(self):
        """Load the supported calls sequentially into one read-only int8 array."""
        packed = np.fromfile(self.packed.path, dtype=np.int8, count=self.packed.shape[0])
        if packed.shape != self.packed.shape:
            raise ValueError("Prepared focal ancestor payload is incomplete")
        packed.flags.writeable = False
        return packed

    def make_readonly(self):
        for field in dataclasses.fields(self):
            array = getattr(self, field.name)
            if isinstance(array, np.ndarray):
                array.flags.writeable = False


@dataclasses.dataclass
class FocalLayout:
    """A panel-ordered catalog whose offsets follow cutoff admission order."""

    catalog: ReferenceCatalog
    admissions: np.ndarray
    prefix_bytes: np.ndarray


@dataclasses.dataclass
class VariantBlock:
    """Reference-site rows selected from one physical variant block."""

    block: int
    local_rows: np.ndarray
    output_rows: np.ndarray


@dataclasses.dataclass
class SampleChunk:
    """Selected source columns and canonical jobs in one physical sample chunk."""

    block: int
    columns: np.ndarray
    job_indices: np.ndarray


@dataclasses.dataclass
class SampleInput:
    """Small, prepared reader adapter; contains no decoded genotype matrix."""

    path: pathlib.Path
    blocks: list[VariantBlock]
    chunks: list[SampleChunk]
    remap: np.ndarray
    allele_mapper: tsinfer.vcz.AlleleMapper
    num_sites: int
    source_rows: np.ndarray
    chunk_shape: tuple[int, ...]
    ancestral: np.ndarray


@dataclasses.dataclass
class SampleSlab:
    """Published only when complete; rows are contiguous complete haplotypes."""

    chunk: int
    calls: MappedCalls | np.ndarray
    job_indices: np.ndarray
    blocks_read: int
    load_s: float


def site_rows(store, positions):
    """Align a sorted store to the reference once, requiring every reference site."""
    source_positions = store["variant_position"][:]
    rows = np.searchsorted(source_positions, positions)
    present = rows < len(source_positions)
    present[present] = source_positions[rows[present]] == positions[present]
    if not np.all(present):
        raise ValueError("Reference sites are absent from a genotype store")
    return rows


def prepare_samples(source, reference, jobs):
    """Resolve columns, identities and native canonical allele maps once."""
    store = tsinfer.vcz.open_store(source.path)
    positions = reference.sites_position.astype(np.int32)
    rows = site_rows(store, positions)
    columns = tsinfer.vcz.resolve_samples_selection(store, source.samples)
    ids = np.asarray(store["sample_id"][:].tolist(), dtype=str)
    genotypes = store["call_genotype"]
    ploidy = genotypes.shape[2]
    expected = [(ids[col], p) for col in columns for p in range(ploidy)]
    actual = [(job.sample_id, job.ploidy_index) for job in jobs]
    if actual != expected or len(set(ids)) != len(ids):
        raise ValueError("Sample identities/order disagree with native jobs")
    ancestral = np.asarray([site.ancestral_state for site in reference.sites()])
    alleles = store["variant_allele"].oindex[rows, :]
    mapper = tsinfer.vcz.build_allele_mapper(
        positions, ancestral, [(positions, alleles)]
    )
    if np.any(mapper.num_alleles_array() != 2):
        raise ValueError("Matching requires biallelic reference sites")
    remap = np.full(alleles.shape, -1, dtype=np.int8)
    site_indices = np.arange(len(positions))
    for code in range(alleles.shape[1]):
        remap[:, code] = mapper.batch_lookup(site_indices, alleles[:, code])
    blocks = []
    variant_blocks = rows // genotypes.chunks[0]
    for block in np.unique(variant_blocks):
        output_rows = np.flatnonzero(variant_blocks == block)
        local_rows = rows[output_rows] - block * genotypes.chunks[0]
        blocks.append(VariantBlock(int(block), local_rows, output_rows))
    chunks = []
    sample_blocks = columns // genotypes.chunks[1]
    for block in np.unique(sample_blocks):
        selected = np.flatnonzero(sample_blocks == block)
        local_columns = columns[selected] - block * genotypes.chunks[1]
        job_indices = selected[:, None] * ploidy + np.arange(ploidy)
        chunks.append(SampleChunk(int(block), local_columns, job_indices.ravel()))
    widths = [len(chunk.job_indices) for chunk in chunks]
    logger.info(
        "Prepared %d haplotypes at %d sites from %d physical sample chunks "
        "(genotype chunk shape %s; selected haplotypes per chunk %d–%d)",
        len(jobs),
        len(rows),
        len(chunks),
        genotypes.chunks,
        min(widths),
        max(widths),
    )
    return SampleInput(
        source.path,
        blocks,
        chunks,
        remap,
        mapper,
        len(rows),
        rows,
        genotypes.chunks,
        ancestral,
    )


def _load_variant_block(genotypes, inputs, chunk, block, slab):
    """Decode one physical tile and write its disjoint slab columns."""
    tile = genotypes.blocks[block.block, chunk.block, :]
    raw = tile[block.local_rows[:, None], chunk.columns[None, :], :]
    del tile
    remap = inputs.remap[block.output_rows]
    safe = np.maximum(raw, 0)
    encoded = remap[np.arange(len(remap))[:, None, None], safe]
    encoded[raw < 0] = -1
    selected = encoded.reshape(len(remap), -1)
    # Sorted reference sites make this block's output rows contiguous.
    # A basic slice writes sample-major spans instead of scattering by site.
    output_start = int(block.output_rows[0])
    output_stop = int(block.output_rows[-1]) + 1
    slab[:, output_start:output_stop] = selected.T
    return genotypes.cdata_shape[2]


def _fill_sample_chunk(inputs, chunk, slab, sample_reader_threads):
    """Decode each required physical tile once into disjoint slab columns."""
    store = tsinfer.vcz.open_store(inputs.path)
    genotypes = store["call_genotype"]
    blocks_read = 0
    if sample_reader_threads == 1:
        for block in inputs.blocks:
            blocks_read += _load_variant_block(genotypes, inputs, chunk, block, slab)
    else:
        blocks = iter(inputs.blocks)
        with cf.ThreadPoolExecutor(max_workers=sample_reader_threads) as executor:
            while batch := list(itertools.islice(blocks, sample_reader_threads)):
                futures = [
                    executor.submit(
                        _load_variant_block, genotypes, inputs, chunk, block, slab
                    )
                    for block in batch
                ]
                for future in futures:
                    blocks_read += future.result()
    return blocks_read


def load_sample_chunk(inputs, chunk, directory, sample_reader_threads=1):
    """Decode a complete sample slab into a temporary mapping."""
    started = time.perf_counter()
    descriptor = MappedCalls(
        directory / f"samples_{chunk.block}.bin",
        (len(chunk.job_indices), inputs.num_sites),
    )
    slab = descriptor.open("w+")
    try:
        blocks_read = _fill_sample_chunk(inputs, chunk, slab, sample_reader_threads)
    finally:
        close_mapping(slab)
    elapsed = time.perf_counter() - started
    return SampleSlab(chunk.block, descriptor, chunk.job_indices, blocks_read, elapsed)


def load_resident_sample_chunk(inputs, chunk, sample_reader_threads=1):
    """Decode and publish one read-only, sample-major array for greedy workers."""
    started = time.perf_counter()
    slab = np.empty((len(chunk.job_indices), inputs.num_sites), dtype=np.int8)
    blocks_read = _fill_sample_chunk(inputs, chunk, slab, sample_reader_threads)
    slab.flags.writeable = False
    elapsed = time.perf_counter() - started
    return SampleSlab(chunk.block, slab, chunk.job_indices, blocks_read, elapsed)


def prepare_reference(panel_path, sample_path, reference, af_cutoff, choice, directory):
    """Select supported spans and assign offsets for a single cutoff."""
    layout = _select_focal_spans(
        panel_path, sample_path, reference, (af_cutoff,), choice, directory
    )
    return layout.catalog


def _select_focal_spans(panel_path, sample_path, reference, cutoffs, choice, directory):
    """Clip support to inference intervals and assign offsets in cutoff order."""
    panel = tsinfer.vcz.open_store(panel_path)
    samples = tsinfer.vcz.open_store(sample_path)
    positions = reference.sites_position.astype(np.int32)
    if not np.array_equal(panel["variant_position"][:], positions):
        raise ValueError("Ancestor panel and reference site axes disagree")
    ancestral = np.asarray([site.ancestral_state for site in reference.sites()])
    if not np.array_equal(panel["variant_allele"][:, 0], ancestral):
        raise ValueError("Ancestor panel and reference polarisation disagree")
    intervals = np.asarray(reference.metadata["sequence_intervals"])
    if not np.array_equal(panel["sequence_intervals"][:], intervals):
        raise ValueError("Ancestor panel and reference inference intervals disagree")
    nodes = {}
    for node in reference.nodes():
        if node.metadata.get("source") == "ancestors":
            nodes[node.metadata["sample_id"]] = node.id
    ids = np.asarray(panel["sample_id"][:].tolist(), dtype=str)
    if len(set(ids)) != len(ids) or not set(ids) <= nodes.keys():
        raise ValueError("Ancestor panel identities disagree with reference nodes")
    focal_positions = panel["sample_focal_positions"][:]
    has_focal = np.any(focal_positions >= 0, axis=1)
    columns = np.flatnonzero(has_focal)
    if choice == "left":
        focal = np.min(
            focal_positions,
            axis=1,
            where=focal_positions >= 0,
            initial=np.iinfo(focal_positions.dtype).max,
        )
    else:
        focal = np.max(focal_positions, axis=1, initial=-1)
    selected_focal = focal[columns]
    selected_seeds = np.searchsorted(positions, selected_focal)
    if np.any(selected_seeds == len(positions)):
        raise ValueError("An ancestor seed is absent from the reference")
    if np.any(positions[selected_seeds] != selected_focal):
        raise ValueError("An ancestor seed is absent from the reference")
    seeds = selected_seeds
    sample_rows = site_rows(samples, positions)
    seed_rows = sample_rows[seeds]
    frequencies = samples["variant_match_eval_derived_af"].oindex[seed_rows]
    thresholds = np.asarray(cutoffs, dtype=frequencies.dtype)
    eligible = frequencies <= thresholds[-1]
    columns = columns[eligible]
    seeds = seeds[eligible]
    admissions = np.searchsorted(thresholds, frequencies[eligible], side="left")
    containing = np.searchsorted(intervals[:, 0], positions[seeds], side="right") - 1
    supports = intervals[containing]
    start_positions = np.maximum(
        panel["sample_start_position"][:][columns], supports[:, 0]
    )
    stop_positions = np.minimum(panel["sample_end_position"][:][columns], supports[:, 1])
    starts = np.searchsorted(positions, start_positions)
    stops = np.searchsorted(positions, stop_positions)
    if np.any(containing < 0) or np.any((seeds < starts) | (seeds >= stops)):
        raise ValueError("An ancestor seed is outside its clipped support")
    lengths = stops - starts
    order = np.argsort(admissions, kind="stable")
    grouped_lengths = lengths[order]
    grouped_ends = np.cumsum(grouped_lengths, dtype=np.int64)
    offsets = np.empty(len(columns), dtype=np.int64)
    offsets[order] = grouped_ends - grouped_lengths
    tier_sizes = np.zeros(len(cutoffs), dtype=np.int64)
    np.add.at(tier_sizes, admissions, lengths)
    prefix_bytes = np.cumsum(tier_sizes)
    descriptor = MappedCalls(directory / "ancestors.bin", (int(prefix_bytes[-1]),))
    parents = np.asarray([nodes[ids[col]] for col in columns], dtype=np.int32)
    catalog = ReferenceCatalog(
        parents, columns, seeds, starts, stops, offsets, descriptor
    )
    catalog.make_readonly()
    return FocalLayout(catalog, admissions, prefix_bytes)


def reference_chunks(panel_path, catalog):
    """One packing job per required physical ancestor-column chunk."""
    panel = tsinfer.vcz.open_store(panel_path)
    width = panel["call_genotype"].chunks[1]
    return np.unique(catalog.columns // width).tolist()


def pack_reference_chunk(genotypes, catalog, chunk, packed):
    """Copy disjoint spans in catalog order, decoding each required tile once.

    Sorted columns bound each job by binary search; numeric support masks keep
    Python copying proportional to actual ancestor/block overlaps.
    """
    height, width, _ = genotypes.chunks
    first = np.searchsorted(catalog.columns, chunk * width)
    last = np.searchsorted(catalog.columns, (chunk + 1) * width)
    ancestors = np.arange(first, last)
    required = set()
    for ancestor in ancestors:
        first = catalog.starts[ancestor] // height
        last = (catalog.stops[ancestor] - 1) // height
        required.update(range(first, last + 1))
    for block in sorted(required):
        tile = genotypes.blocks[block, chunk, 0]
        block_start = block * height
        _pack_tile(
            tile,
            packed,
            catalog.columns,
            catalog.starts,
            catalog.stops,
            catalog.offsets,
            ancestors,
            block_start,
            chunk * width,
        )
        del tile
    seed_offsets = catalog.offsets[ancestors]
    seed_offsets = seed_offsets + catalog.seeds[ancestors] - catalog.starts[ancestors]
    if np.any(packed[seed_offsets] != 1):
        raise ValueError("An eligible ancestor seed is not called and derived")
    return len(required)


@numba.njit(cache=True, nogil=True)
def _pack_tile(
    tile, packed, columns, starts, stops, offsets, ancestors, block_start, column_start
):
    """Copy only supported spans while releasing the GIL for other readers."""
    block_stop = block_start + tile.shape[0]
    for ancestor in ancestors:
        start = max(starts[ancestor], block_start)
        stop = min(stops[ancestor], block_stop)
        column = columns[ancestor] - column_start
        offset = offsets[ancestor] + start - starts[ancestor]
        for site in range(start, stop):
            packed[offset + site - start] = tile[site - block_start, column, 0]


def _cutoff_group(cutoffs):
    return tuple(sorted({float(cutoff) for cutoff in cutoffs}))


def focal_directory(reference_path, choice, cutoffs):
    """Destination shared by all greedy jobs in a configured cutoff group."""
    cutoffs = _cutoff_group(cutoffs)
    group_id = str(cutoffs[0])
    if len(cutoffs) > 1:
        group_id = hashlib.sha256(repr(cutoffs).encode()).hexdigest()
    return reference_path.parent / "focal" / f"{reference_path.stem}_{choice}_{group_id}"


def _fingerprint_path(digest, path):
    """Fingerprint local file identities without reading genotype tiles."""
    resolved = path.resolve(strict=True)
    digest.update(os.fsencode(resolved))
    if resolved.is_dir():
        for root, directories, files in os.walk(resolved, followlinks=True):
            directories.sort()
            for name in sorted(files):
                entry = pathlib.Path(root, name)
                stat = entry.stat()
                identity = (
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
                digest.update(os.fsencode(entry.relative_to(resolved)))
                digest.update(repr(identity).encode())
    else:
        stat = resolved.stat()
        digest.update(
            repr(
                (
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
            ).encode()
        )


def _focal_key(panel_path, sample_path, reference_path, choice, cutoffs):
    # Use the original key for products with one tier.
    settings = cutoffs[0] if len(cutoffs) == 1 else cutoffs
    digest = hashlib.sha256(repr((choice, settings)).encode())
    _fingerprint_path(digest, panel_path)
    for name in ("zarr.json", "variant_position", "variant_match_eval_derived_af"):
        path = sample_path / name
        if path.exists():
            _fingerprint_path(digest, path)
    _fingerprint_path(digest, reference_path)
    return digest.hexdigest()


def _write_focal_reference(panel_path, layout, workers):
    """Write disjoint spans directly into the product's only genotype buffer."""
    catalog = layout.catalog
    genotypes = tsinfer.vcz.open_store(panel_path)["call_genotype"]
    chunks = np.unique(catalog.columns // genotypes.chunks[1])
    catalog.packed.path.touch()
    packed = catalog.packed.open("w+")
    try:
        pack = functools.partial(pack_reference_chunk, genotypes, catalog, packed=packed)
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            tiles = sum(pool.map(pack, chunks))
        if isinstance(packed, np.memmap):
            packed.flush()
    finally:
        close_mapping(packed)
    fields = (
        field.name for field in dataclasses.fields(catalog) if field.name != "packed"
    )
    arrays = {name: getattr(catalog, name) for name in fields}
    if len(layout.prefix_bytes) > 1:
        arrays.update(admissions=layout.admissions, prefix_bytes=layout.prefix_bytes)
    np.savez(catalog.packed.path.with_name("catalog.npz"), **arrays)
    return tiles


def _read_focal_catalog(product, tier):
    """Read catalog metadata in panel order and describe the matching byte prefix."""
    fields = (
        field.name
        for field in dataclasses.fields(ReferenceCatalog)
        if field.name != "packed"
    )
    with np.load(product / "catalog.npz") as saved:
        if "admissions" in saved.files:
            eligible = saved["admissions"] <= tier
            arrays = {name: saved[name][eligible] for name in fields}
            size = int(saved["prefix_bytes"][tier])
        else:
            arrays = {name: saved[name] for name in fields}
            size = int(np.sum(arrays["stops"] - arrays["starts"]))
    payload = product / "ancestors.bin"
    if payload.stat().st_size < size:
        raise ValueError("Prepared focal ancestor payload is incomplete")
    catalog = ReferenceCatalog(**arrays, packed=MappedCalls(payload, (size,)))
    catalog.make_readonly()
    return catalog


def prepare_focal_reference(
    panel_path,
    sample_path,
    reference_path,
    reference,
    af_cutoff,
    choice,
    workers,
    collector=None,
    *,
    af_cutoffs=None,
):
    """Publish the catalog and payload together and return an immutable catalog.

    Inputs are immutable during a workflow job. Recheck their identities before
    publishing so a replaced input cannot leave a product under its previous key.
    """
    af_cutoff = float(af_cutoff)
    panel_path = pathlib.Path(panel_path)
    sample_path = pathlib.Path(sample_path)
    if af_cutoffs is None:
        af_cutoffs = (af_cutoff,)
    cutoffs = _cutoff_group(af_cutoffs)
    tier = cutoffs.index(af_cutoff)
    directory = focal_directory(reference_path, choice, cutoffs)
    directory.mkdir(parents=True, exist_ok=True)
    tiles = 0
    started = time.perf_counter()
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        key = _focal_key(panel_path, sample_path, reference_path, choice, cutoffs)
        product = directory / key
        converted = not product.is_dir()
        if converted:
            with performance.measure(collector, "reference_conversion"):
                with tempfile.TemporaryDirectory(
                    dir=directory, prefix=".packing_"
                ) as tmp:
                    staging = pathlib.Path(tmp) / "product"
                    staging.mkdir()
                    layout = _select_focal_spans(
                        panel_path, sample_path, reference, cutoffs, choice, staging
                    )
                    tiles = _write_focal_reference(panel_path, layout, workers)
                    verified_key = _focal_key(
                        panel_path, sample_path, reference_path, choice, cutoffs
                    )
                    if verified_key != key:
                        raise RuntimeError(
                            "Focal reference inputs changed during conversion"
                        )
                    staging.rename(product)
    catalog = _read_focal_catalog(product, tier)
    elapsed = time.perf_counter() - started
    if collector is not None:
        collector.count(ancestor_blocks_read=tiles)
        collector.diagnostics["focal_reference_converted"] = converted
    logger.info(
        "%s focal reference in %.3fs (%d genotype tiles, %d useful bytes): %s",
        "Converted" if converted else "Opened",
        elapsed,
        tiles,
        catalog.packed.shape[0],
        product,
    )
    return catalog
