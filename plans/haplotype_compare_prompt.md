For what follows, restrict the analysis to inferred ancestors. The
tsinfer-match-eval pipeline stores ancestral haplotypes as a zarr, along with a
data structure that shows how the ancestors that have focal mutations carried by
a sample for each sample. There are frequency cutoffs that select subsets of
ancestors by focal frequency. The pipeline already analyses the paths that
matching outputs, but I want to do next is to not use the ancestor TS or the HMM
at all. Instead, for each sample and max cutoff I want to load the ancestral
haplotypes and determine the bits of them that match the sample perfectly with
up to K mismatches.

Review ~/exo/documents/meeting_notes/2026-09-10 (Meeting with Gil).md under
Matching-engine performance for how my supervisor explained it, because that's
what we will implement. For performance, we should only have to do it once for
each sample with the maximum frequency cutoff. First step is to get the sample
haplotype in the same coordinate system as the ancestors if it isn't already (it
won't likely because of singletons). Here is an efficient approach to do what he
suggests.

We start with the list of focal sites with focal ancestors below the cutoff and
the streamable, aligned array of all the ancestral haplotypes with the sample
haplotype. We sweep from left to right first, streaming in the genotypes as we
go without needing to keep all of them in memory. As soon as we hit a focal site
we start tracking comparing its focal ancestor haplotype to the sample
haplotype, storing the site index of each mismatch until we reach K mismatches
then store the site index and stop tracking that ancestor in memory (this gives
us all the answers for <= K too without extra effort required) The stored site
indices are the right bounds for that ancestor for each mismatch count. We keep
adding and clearing ancestors as we sweep left to right until we have determined
the right bounds for all the ancestor chunks. Then we go backwards from end to
start and do the same for the left bounds with the same approach (should be able
to reuse code with a direction or something?). By the end we have a dataset of
all the left, right site indices for every ancestor and each k <= K (we should
store the focal site index here too). Include the focal AC in this dataframe
too, since we can use this to filter out chunks by cutoff continuously. I think
we want this algorithm in numba for speed. We should end up with a big dataframe
(should be small enough to fit) for all the samples, max_mismatches (what I'd
call k in the dataframe, focal_ac, focal_af and left and right bounds. Store the
positions and bounds in bp too using a sites_position lookup. This should give
us all we need to test my supervisors idea. Note that we don't need the
haplotypes at all here, just bounds. Not sure how best to handle the genotype
streaming so I leave this to you to figure out.

Call the rule construct_focal_ancestor_chunks and make its output the big
dataframe. Put the main function(s) it uses in lib/haplotypes.py. Test it with
the n300 1mb data until finalised, then make the dataframe for the n100 10mb
data too. All this should go in a detailed plans/haplotype_compare.md plan, you
can do some read only testing but don't change other files. Keep the approach
clean and simple, no defensive coding or unnecessarily convoluted classes and
data structures.
