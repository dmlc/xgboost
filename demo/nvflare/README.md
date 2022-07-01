# Experimental Support of Federated XGBoost using NVFlare

This directory contains a demo of Federated Learning using
[NVFlare](https://nvidia.github.io/NVFlare/).

## Training with CPU only

To run the demo, first build XGBoost with the federated learning plugin enabled (see the
[README](../../plugin/federated/README.md)).

Install NVFlare (note that currently NVFlare only supports Python 3.8; for NVFlare 2.1.2 we also
need to pin the protobuf package to 3.20.x to avoid protoc errors):
```shell
pip install nvflare protobuf==3.20.1
```

Prepare the data:
```shell
./prepare_data.sh
```

Start the NVFlare federated server:
```shell
./poc/server/startup/start.sh
```

In another terminal, start the first worker:
```shell
./poc/site-1/startup/start.sh
```

And the second worker:
```shell
./poc/site-2/startup/start.sh
```

Then start the admin CLI, using `admin/admin` as username/password:
```shell
./poc/admin/startup/fl_admin.sh
```

In the admin CLI, run the following command:
```shell
submit_job hello-xgboost
```

Once the training finishes, the model file should be written into
`./poc/site-1/run_1/test.model.json` and `./poc/site-2/run_1/test.model.json`
respectively.

Finally, shutdown everything from the admin CLI:
```shell
shutdown client
shutdown server
```

## Training with GPUs

To demo with Federated Learning using GPUs, make sure your machine has at least 2 GPUs.
Build XGBoost with the federated learning plugin enabled along with CUDA, but with NCCL
turned off (see the [README](../../plugin/federated/README.md)).

Modify `config/config_fed_client.json` and set `use_gpus` to `true`, then repeat the steps
above.
