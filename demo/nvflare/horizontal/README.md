# Experimental Support of Horizontal Federated XGBoost using NVFlare

This directory contains a demo of Horizontal Federated Learning using
[NVFlare](https://nvidia.github.io/NVFlare/).

## Training with CPU only

To run the demo, first build XGBoost with the federated learning plugin enabled (see the
[README](../../plugin/federated/README.md)).

Install NVFlare (note that currently NVFlare only supports Python 3.8):
```shell
pip install nvflare
```

Prepare the data:
```shell
./prepare_data.sh
```

Start the NVFlare federated server:
```shell
/tmp/nvflare/poc/server/startup/start.sh
```

In another terminal, start the first worker:
```shell
/tmp/nvflare/poc/site-1/startup/start.sh
```

And the second worker:
```shell
/tmp/nvflare/poc/site-2/startup/start.sh
```

Then start the admin CLI:
```shell
/tmp/nvflare/poc/admin/startup/fl_admin.sh
```

In the admin CLI, run the following command:
```shell
submit_job horizontal-xgboost
```

Once the training finishes, the model file should be written into
`/tmp/nvlfare/poc/site-1/run_1/test.model.json` and `/tmp/nvflare/poc/site-2/run_1/test.model.json`
respectively.

Finally, shutdown everything from the admin CLI, using `admin` as password:
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
