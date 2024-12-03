## Example input:
##   xgb-ci.gpu_build_r_rockylinux8
## Example output:
##   --build-arg CUDA_VERSION_ARG=12.4.1 --build-arg R_VERSION_ARG=4.3.2
def compute_build_args($input; $container_id):
  $input |
  .[$container_id] |
  select(.build_args != null) |
  .build_args |
  to_entries |
  map("--build-arg " + .key + "=" + .value) |
  join(" ");
