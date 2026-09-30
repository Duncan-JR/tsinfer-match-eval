# Implement matching validation

Review the thread 11:57 Design matching evaluation pipeline in
~/exo/this_week.org for context along with
~/exo/documents/meeting_notes/2026-09-10 (Meeting with Gil).md starting from ###
General pruning/indexing idea. Also look at the latest commit in ~/work/tsinfer
for guidance on how to do the ancestor matching and sample matching separately,
paying attention to what is needed to ensure the correct sequence_length is
used.

- Add match_ancestors step that constructs ancestor TS from collected ancestors.
  write_inference_config should do everything you need to run the command for
  inferred ancestors, you just need to be sure to configure it correctly. I
  think you'll have to make an optional write_true_ancestors_config step to make
  the toml for running ancestor matching on the true ancestor zarr (optional
  because there might not be any simulated data). Use default HMM parameters for
  now - I know they aren't necessarily correct for recombination maps and such
  but that's fine. Disable path compression.

- Add find_focal_ancestors step that finds all the sites with 1s in each sample
  (i.e. the alleles they carry) and looks up the corresponding ancestors with
  those mutations as focal in the ancestor dataframes; both true and inferred if
  true is present, just inferred otherwise. Not sure how best to store this data
  since each sample will have a variable number of focal ancestors: figure
  something out that is simple and efficient preferably one file for each true
  and inferred ancestor zarr.

- Add match_samples step that collects all the ancestors ts and matches the
  samples against them from the zarr. This step should output the
  un-post-processed TS and all the data from the match_file for each sample.
  Again don't path compress or post process and use default HMM parameters. If
  there is a path likelihood in the match_file definitely store that - if not,
  see if there is a way to get it out because it will be important later. Again
  I'm not sure how best to store the match_file data efficiently so I leave that
  to you to figure out, but we ideally don't want one file per sample but rather
  one file per ancestor zarr. Importantly we don't want to compute anything with
  the focal_ancestors here, this is just doing sample matching and outputing all
  the useful data from it.

The next step will use the find_focal_ancestors and match_samples match_files
data to compute interesting properties, but that is out of scope for this plan.
Write an implementation plan to execute only the above steps, with rule all
collecting the focal ancestors data, inferred ancestors TS and `match_file` data
for true and inferred ancestors. Keep to the style of this repo with clean,
simple code that does the bare minimum to get it working without elaborate
checks, premature optimisations or unnecessarily complex data structures. I want
code that's easy to read and understand - we can worry about optimisation later,
we just need to choose a reasonable approach. Your plan should show the names of
steps, their inputs and outputs, and associated functions that will be needed to
run them and their locations, highlighting data structures chosen and crucial
algorithms. Matching-related functions should go in lib/matching.py, including
find_focal_ancestors stuff I suppose. Misc. functions should go in lib/utils.py.
Make it in plans/matching_setup.md and don't change other files.

There shouldn't be any unit tests made but the executor of the plan will
certainly need to check it works with the previous plan (initial_mvp.md) example
data. Don't plan any validation with real datasets yet.
