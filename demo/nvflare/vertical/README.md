# Experimental Support of Vertical Federated XGBoost using NVFlare

This directory contains a demo of Vertical Federated Learning using
[NVFlare](https://nvidia.github.io/NVFlare/).

## Training with CPU only

To run the demo, first build XGBoost with the federated learning plugin enabled (see the
[README](../../plugin/federated/README.md)).

Install NVFlare (note that currently NVFlare only supports Python 3.8):
```shell
pip install nvflare
```

Prepare the data (note that this step will download the HIGGS dataset, which is 2.6GB compressed, and 7.5GB
uncompressed, so make sure you have enough disk space and are on a fast internet connection):
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
submit_job vertical-xgboost
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

Currently GPUs are not yet supported by vertical federated XGBoost.
