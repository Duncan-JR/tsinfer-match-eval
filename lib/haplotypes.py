"""Packed ancestor references and physical-block, selected-cohort sample loading."""

import dataclasses
import logging
import pathlib
import time

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
        source.path, blocks, chunks, remap, mapper, len(rows), rows, genotypes.chunks
    )


def load_sample_chunk(inputs, chunk, directory):
    """Read each required physical tile once, selecting before numeric remapping."""
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
        for block in inputs.blocks:
            # Slice all ploidy blocks together, keeping both chromosome copies.
            tile = genotypes.blocks[block.block, chunk.block, :]
            raw = tile[block.local_rows[:, None], chunk.columns[None, :], :]
            del tile
            remap = inputs.remap[block.output_rows]
            safe = np.maximum(raw, 0)
            encoded = remap[np.arange(len(remap))[:, None, None], safe]
            encoded[raw < 0] = -1
            selected = encoded.reshape(len(remap), -1)
            slab[:, block.output_rows] = selected.T
            blocks_read += genotypes.cdata_shape[2]
    finally:
        close_mapping(slab)
    elapsed = time.perf_counter() - started
    return SampleSlab(chunk.block, descriptor, chunk.job_indices, blocks_read, elapsed)


def prepare_reference(panel_path, sample_input, reference, af_cutoff, choice, directory):
    """Select by cohort-derived seed AF and clip supports before any extension.

    Genotypes are packed by :func:`pack_reference_chunk` in the persistent pool.
    A shared seed does not collapse distinct panel ancestors.
    """
    panel = tsinfer.vcz.open_store(panel_path)
    samples = tsinfer.vcz.open_store(sample_input.path)
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
    seeds = np.full(len(ids), -1, dtype=np.int64)
    for column, focal_positions in enumerate(panel["sample_focal_positions"][:]):
        focal_positions = focal_positions[focal_positions >= 0]
        if len(focal_positions) > 0:
            focal = min(focal_positions) if choice == "left" else max(focal_positions)
            seed = np.searchsorted(positions, focal)
            if seed == len(positions) or positions[seed] != focal:
                raise ValueError("An ancestor seed is absent from the reference")
            seeds[column] = seed
    columns = np.flatnonzero(seeds >= 0)
    sample_rows = sample_input.source_rows
    seed_rows = sample_rows[seeds[columns]]
    frequencies = samples["variant_match_eval_derived_af"].oindex[seed_rows]
    columns = columns[frequencies <= af_cutoff]
    seeds = seeds[columns]
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
    packed = descriptor.open("w+")
    close_mapping(packed)
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


def pack_reference_chunk(panel_path, catalog, chunk):
    """Copy disjoint spans in catalog order, decoding each required tile once.

    Sorted columns bound each job by binary search; numeric support masks keep
    Python copying proportional to actual ancestor/block overlaps.
    """
    panel = tsinfer.vcz.open_store(panel_path)
    genotypes = panel["call_genotype"]
    height, width, _ = genotypes.chunks
    first = np.searchsorted(catalog.columns, chunk * width)
    last = np.searchsorted(catalog.columns, (chunk + 1) * width)
    ancestors = np.arange(first, last)
    starts = catalog.starts[first:last]
    stops = catalog.stops[first:last]
    required = set()
    for ancestor in ancestors:
        first = catalog.starts[ancestor] // height
        last = (catalog.stops[ancestor] - 1) // height
        required.update(range(first, last + 1))
    packed = catalog.packed.open("r+")
    try:
        for block in sorted(required):
            tile = genotypes.blocks[block, chunk, 0]
            block_start = block * height
            block_stop = block_start + tile.shape[0]
            overlapping = (starts < block_stop) & (stops > block_start)
            for ancestor in ancestors[overlapping]:
                start = max(catalog.starts[ancestor], block_start)
                stop = min(catalog.stops[ancestor], block_stop)
                column = catalog.columns[ancestor] - chunk * width
                offset = catalog.offsets[ancestor] + start - catalog.starts[ancestor]
                packed[offset : offset + stop - start] = tile[
                    start - block_start : stop - block_start, column, 0
                ]
            del tile
        for ancestor in ancestors:
            offset = catalog.offsets[ancestor]
            seed_offset = catalog.seeds[ancestor] - catalog.starts[ancestor]
            if packed[offset + seed_offset] != 1:
                raise ValueError("An eligible ancestor seed is not called and derived")
    finally:
        close_mapping(packed)
    return len(required)
