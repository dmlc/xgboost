def compute_build_args($input; $container_id):
  $input |
  .[$container_id].build_args |
  to_entries |
  map("--build-arg " + .key + "=" + .value) |
  join(" ");
