We've yet to setup analysis of real datasets using this pipeline, but in theory
it shouldn't be too tricky. The example I want to test with is the
~/work/dphil-analysis/data/zarr_vcfs/tgp/chr20/data.zarr/ 1000 genomes project
data. Key features we need to add to the pipeline is to support masking of sites
and samples. Ideally this should be specified in the pipeline yaml and thus
incorporated into the toml used by tsinfer. In the yaml I would want to specify
genomic regions with include like in the tsinfer example toml which is also
based on 1000 genomes (review this carefully along with the details of filtering
in tsinfer). For testing, start with the 1 mbp region from 1e6 to 2e6. In
addition, we need to be able to specify site masks and sample masks that we
assume exist in the zarr already for simplicity. In the test case, you should
use sample_1kgp_100_subset_mask for the samples (true for masked out samples)j
and
~/work/dphil-analysis/data/zarr_vcfs/tgp/chr20/data.zarr/variant_1kgp_100_subset_chr20p_region_filterNton23_site_density_threshold_sites_per_kbp_5_window_size_100000_mask/
for the site_mask, both of which should be optionally specified in the yaml (in
general the user might not need to specify and masks or include filters, in
which case the entire zarr is used). Make a detailed implementation plan
(plans/real_data.md) to fully support real data sets like this one with the
features I discuss. The plan should ensure that the entire pipeline runs to
completion given the filters and masks I specified above but still not make any
unit tests. Don't change code but you may run read-only analyses as needed to
form the plan. Keep the approach simple and avoid elaborate workarounds: this
shouldn't require lots of code to do.
