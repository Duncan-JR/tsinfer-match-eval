"""Small, importable helpers for the greedy orchestration smoke checks."""

import dataclasses
import pathlib
import sys
import time

import numpy as np
import tsinfer
import tskit
import zarr

from lib import benchmark, haplotypes, matching, performance, utils


def canonical(path):
    """Compare identities, squashed parent runs and sample mutations without rows."""
    ts = tskit.load(path)
    result = {}
    paths = {}
    changes = {}
    for edge in ts.edges():
        paths.setdefault(edge.child, []).append((edge.left, edge.right, edge.parent))
    for mutation in ts.mutations():
        changes.setdefault(mutation.node, []).append(
            (ts.site(mutation.site).position, mutation.derived_state)
        )
    for node in ts.nodes():
        if node.metadata.get("source") != "samples":
            continue
        identity = tuple(
            node.metadata[key] for key in ("source", "sample_id", "ploidy_index")
        )
        edges = sorted(paths.get(node.id, []))
        squashed = []
        for left, right, parent in edges:
            if squashed and squashed[-1][1] == left and squashed[-1][2] == parent:
                squashed[-1] = (squashed[-1][0], right, parent)
            else:
                squashed.append((left, right, parent))
        mutations = sorted(changes.get(node.id, []))
        result[identity] = (squashed, mutations)
    return result


@dataclasses.dataclass
class Tiny:
    root: pathlib.Path
    config: pathlib.Path
    reference: pathlib.Path
    panel: pathlib.Path
    source: pathlib.Path
    calls: np.ndarray
    columns: np.ndarray


def make_tiny(root, settings, hmm):
    """Create three selected physical chunks, missing calls and mixed polarity."""
    root.mkdir(exist_ok=True)
    sites = settings["variants"]
    individuals = settings["individuals"]
    positions = np.arange(1, sites + 1, dtype=np.int32) * 10
    calls = np.random.default_rng(settings["seed"]).integers(
        0, 2, size=(sites, individuals, 2), dtype=np.int8
    )
    calls[0, 0, 0] = -1
    calls[2, 4, 1] = -1
    calls[-1] = -1
    alleles = np.tile(np.array(["A", "T"]), (sites, 1))
    ancestral = alleles[np.arange(sites), np.arange(sites) % 2]
    source = root / "input.zarr"
    store = zarr.open_group(source, mode="w", zarr_format=3)
    arrays = {
        "call_genotype": (
            calls,
            settings["genotype_chunks"],
            ["variants", "samples", "ploidy"],
        ),
        "variant_position": (positions, [4], ["variants"]),
        "variant_allele": (alleles, [4, 2], ["variants", "alleles"]),
        "variant_ancestral_state": (ancestral, [4], ["variants"]),
        "sample_id": (np.array([f"s{i}" for i in range(individuals)]), [2], ["samples"]),
        "variant_contig": (np.zeros(sites, dtype=np.int8), [4], ["variants"]),
        "contig_id": (np.array(["chr1"]), [1], ["contigs"]),
        "contig_length": (np.array([int(positions[-1] + 10)]), [1], ["contigs"]),
    }
    for name, (values, chunks, dimensions) in arrays.items():
        array = store.create_array(name, data=values, chunks=tuple(chunks))
        array.attrs["_ARRAY_DIMENSIONS"] = dimensions
    selected = settings["selected_samples"]
    selection = ",".join(selected)
    columns = np.array([int(sid[1:]) for sid in selected])
    masked = root / "samples.zarr"
    utils.annotate_derived_counts(
        source, masked, {"field": "variant_ancestral_state"}, selection
    )
    # The fixture has three eligible ancestors, spanning several physical blocks.
    panel = root / "ancestors.zarr"
    canonical_alleles = alleles.copy()
    canonical_alleles[1::2] = canonical_alleles[1::2, ::-1]
    panel_calls = np.zeros((sites, 3, 1), dtype=np.int8)
    panel_calls[1:6:2, 0, 0] = 1
    panel_calls[2:6:2, 1, 0] = 1
    panel_calls[7:12:2, 2, 0] = 1
    intervals = np.array([[0, positions[6]], [positions[6], positions[-1] + 10]])
    height, width, _ = settings["panel_chunks"]
    ancestors = tsinfer.vcz.setup_ancestor_zarr(
        sites,
        positions,
        canonical_alleles,
        np.zeros(sites, dtype=np.int8),
        intervals,
        store=panel,
        samples_chunk_size=width,
        variants_chunk_size=height,
        contig_id="chr1",
        contig_length=int(positions[-1] + 10),
    )
    ancestors["call_genotype"].resize(panel_calls.shape)
    ancestors["call_genotype"][:] = panel_calls
    for name, values in {
        "sample_time": [3, 2, 1],
        "sample_start_position": [10, 10, 10],
        "sample_end_position": [
            int(positions[4]),
            int(positions[-1] + 10),
            int(positions[-1] + 10),
        ],
    }.items():
        ancestors[name].resize((3,))
        ancestors[name][:] = values
    tsinfer.vcz.finalize_ancestor_zarr(ancestors, [[20], [30], [80]])
    cfg_path = root / "native.toml"
    utils.make_inference_config(
        masked,
        panel,
        cfg_path,
        {"field": "variant_ancestral_state"},
        hmm,
        sample_selection=selection,
    )
    reference = panel.with_suffix(".trees")
    cfg = matching._sample_config(cfg_path, reference, root / "unused.trees")
    jobs, individuals_md, populations_md = tsinfer.pipeline._build_jobs_with_metadata(
        cfg, positions
    )
    tables = tskit.TableCollection(float(positions[-1] + 10))
    tables.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.metadata = {"sequence_intervals": intervals.tolist()}
    for table in (tables.nodes, tables.individuals, tables.populations):
        table.metadata_schema = tskit.MetadataSchema.permissive_json()
    for metadata in individuals_md:
        tables.individuals.add_row(metadata=metadata)
    for metadata in populations_md:
        tables.populations.add_row(metadata=metadata)
    tables.nodes.add_row(time=6)
    tables.nodes.add_row(time=5)
    tables.edges.add_row(0, tables.sequence_length, 0, 1)
    for ancestor in range(3):
        node = tables.nodes.add_row(
            time=3 - ancestor,
            metadata={
                "source": "ancestors",
                "sample_id": f"a{ancestor}",
                "ploidy_index": 0,
            },
        )
        tables.edges.add_row(0, tables.sequence_length, 1, node)
    for site, position in enumerate(positions):
        tables.sites.add_row(position, ancestral[site])
        for ancestor in np.flatnonzero(panel_calls[site, :, 0] == 1):
            tables.mutations.add_row(site, int(ancestor + 2), canonical_alleles[site, 1])
    tables.sort()
    tables.tree_sequence().dump(reference)
    return Tiny(root, cfg_path, reference, panel, masked, calls, columns)


def prepared(tiny, directory):
    reference = tskit.load(tiny.reference)
    cfg = matching._sample_config(
        tiny.config, tiny.reference, directory / "unused.trees"
    )
    jobs, _, _ = tsinfer.pipeline._build_jobs_with_metadata(
        cfg, reference.sites_position
    )
    jobs.sort(key=lambda job: job.haplotype_index)
    samples = haplotypes.prepare_samples(cfg.sources["samples"], reference, jobs)
    return reference, jobs, samples


def delayed_trial(spec_path, delay):
    """Importable validation delay used only to exercise the real trial launcher."""
    original = benchmark.quality_check

    def validate(*args):
        time.sleep(delay)
        return original(*args)

    benchmark.quality_check = validate
    with performance.sample_memory():
        return benchmark.trial(spec_path)


if __name__ == "__main__":
    delayed_trial(pathlib.Path(sys.argv[1]), float(sys.argv[2]))


def frozen_pack_reference(panel_path, catalog, chunk):
    """Packing before the binary-search/overlap change, frozen for byte comparisons."""
    panel = tsinfer.vcz.open_store(panel_path)
    genotypes = panel["call_genotype"]
    height, width, _ = genotypes.chunks
    ancestors = np.flatnonzero(catalog.columns // width == chunk)
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
            for ancestor in ancestors:
                start = max(catalog.starts[ancestor], block_start)
                stop = min(catalog.stops[ancestor], block_start + tile.shape[0])
                if start < stop:
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
        haplotypes.close_mapping(packed)
    return len(required)


def pack_job(task):
    panel, catalog, chunk = task
    return haplotypes.pack_reference_chunk(panel, catalog, chunk)


class TrackedOffsets(np.ndarray):
    """Count Python scalar copying/seed visits without timing instrumentation."""

    def __new__(cls, values):
        result = np.asarray(values).view(cls)
        result.visits = 0
        return result

    def __getitem__(self, index):
        if np.isscalar(index):
            self.visits += 1
        return super().__getitem__(index)
