# Matching evaluation

We can now need to integrate the focal ancestors and match results data to
answer some key questions about what the HMM is doing. We want to know how often
the focal ancestors are being picked as the best matching parent to samples with
those focal ancestors. Not only that, the key question is how often *recent*
focal ancestors are a good match; e.g., if we only use doubleton and tripleton
focal ancestors. 

Lets make a configured `ac_cutoff` parameter in the config and set it initially to
have values 3, 5,10,100,600. This keys into the derived_af of the ancestors
dataframe, giving us increasingly large subsets of ancestors with focal
frequency <= cutoff = f (i.e up to tripletons, up to five-tons, etc.).

For each sample, we know what parents they have been matched to for which
genomic regions. For each cutoff bin, I want to know:

- Fraction of evaluated bases copied from focal ancestors
- Fraction of sites copied from focal ancestors (converted from bp result using
  sites_position)
- Number of switches on path (fixed for all f)
- Number of mismatches on path (fixed for all f)

At this point this is a diagnostic using data we already have, although for
efficiency we might need to reorganise data to handle the multiple cutoffs
efficiently. If you think I'm missing useful variables feel free to suggest more.

The output of the `compute_focal_ancestor_stats` method should be a dataframe with
`num_cutoffs*num_samples` rows with the above data. Here we need to be mindful
about performance: I suspect we shouldn't need to run through the match results
for each cutoff but rather just do it once for the maximum cutoff for each
sample (in this case, all the focal ancestors for that sample). I'm hoping we
can build a datastructure that can be adjusted for the lower frequencies if we
have easy access to the set of ancestor IDs of focal ancestors below each
cutoff.

Make plans/matching_eval.md to plan the implementation of this idea thoroughly
in the pipeline. You can reorganise the existing data structures as needed to
make this efficient. Numba is an option here if it helps, as long as the code
stays simple, easy to follow and not overly convoluted. In the plan, all
functions, steps, inputs and outputs and their naming schemes should be planned
along with the internal testing needed to ensure it's implemented correctly.
Don't edit any files besides this but you can run read only experiments if need
be to test ideas.
