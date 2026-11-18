"""Packed ancestor references and physical-block, selected-cohort sample loading."""

import concurrent.futures as cf
import dataclasses
import itertools
import logging
import pathlib
import time

import numba
import numpy as np
import tsinfer

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

    def make_readonly(self):
        for field in dataclasses.fields(self):
            array = getattr(self, field.name)
            if isinstance(array, np.ndarray):
                array.flags.writeable = False


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
    calls: MappedCalls
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


def load_sample_chunk(inputs, chunk, directory, sample_reader_threads=1):
    """Read each physical tile once; publish only after every block finishes."""
    started = time.perf_counter()
    store = tsinfer.vcz.open_store(inputs.path)
    genotypes = store["call_genotype"]
    descriptor = MappedCalls(
        directory / f"samples_{chunk.block}.bin",
        (len(chunk.job_indices), inputs.num_sites),
    )
    slab = descriptor.open("w+")
    blocks_read = 0
    try:
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
    finally:
        close_mapping(slab)
    elapsed = time.perf_counter() - started
    return SampleSlab(chunk.block, descriptor, chunk.job_indices, blocks_read, elapsed)


def prepare_reference(panel_path, sample_input, reference, af_cutoff, choice, directory):
    """Select by cohort-derived seed AF and clip supports before any extension.

    The caller owns storage for :func:`pack_reference_chunk`.
    A shared seed does not collapse distinct panel ancestors.
    """
    panel = tsinfer.vcz.open_store(panel_path)
    samples = tsinfer.vcz.open_store(sample_input.path)
    positions = reference.sites_position.astype(np.int32)
    if not np.array_equal(panel["variant_position"][:], positions):
        raise ValueError("Ancestor panel and reference site axes disagree")
    if not np.array_equal(panel["variant_allele"][:, 0], sample_input.ancestral):
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
    sample_rows = sample_input.source_rows
    seed_rows = sample_rows[seeds]
    frequencies = samples["variant_match_eval_derived_af"].oindex[seed_rows]
    eligible = frequencies <= af_cutoff
    columns = columns[eligible]
    seeds = seeds[eligible]
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
    offsets = np.zeros(len(columns), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    descriptor = MappedCalls(directory / "ancestors.bin", (int(lengths.sum()),))
    parents = np.asarray([nodes[ids[col]] for col in columns], dtype=np.int32)
    catalog = ReferenceCatalog(
        parents, columns, seeds, starts, stops, offsets, descriptor
    )
    catalog.make_readonly()
    return catalog


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
