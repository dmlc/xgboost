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

Make a note of the job id:
```console
Submitted job: 28309e77-a7c5-45e6-b2bc-c2e3655122d8
```

On both workers, you should see train and eval losses printed:
```console
[10:45:41] [0]	eval-logloss:0.22646	train-logloss:0.23316
[10:45:41] [1]	eval-logloss:0.13776	train-logloss:0.13654
[10:45:41] [2]	eval-logloss:0.08036	train-logloss:0.08243
[10:45:41] [3]	eval-logloss:0.05830	train-logloss:0.05645
[10:45:41] [4]	eval-logloss:0.03825	train-logloss:0.04148
[10:45:41] [5]	eval-logloss:0.02660	train-logloss:0.02958
[10:45:41] [6]	eval-logloss:0.01386	train-logloss:0.01918
[10:45:41] [7]	eval-logloss:0.01018	train-logloss:0.01331
[10:45:41] [8]	eval-logloss:0.00847	train-logloss:0.01112
[10:45:41] [9]	eval-logloss:0.00691	train-logloss:0.00662
[10:45:41] [10]	eval-logloss:0.00543	train-logloss:0.00503
[10:45:41] [11]	eval-logloss:0.00445	train-logloss:0.00420
[10:45:41] [12]	eval-logloss:0.00336	train-logloss:0.00355
[10:45:41] [13]	eval-logloss:0.00277	train-logloss:0.00280
[10:45:41] [14]	eval-logloss:0.00252	train-logloss:0.00244
[10:45:41] [15]	eval-logloss:0.00177	train-logloss:0.00193
[10:45:41] [16]	eval-logloss:0.00156	train-logloss:0.00161
[10:45:41] [17]	eval-logloss:0.00135	train-logloss:0.00142
[10:45:41] [18]	eval-logloss:0.00123	train-logloss:0.00125
[10:45:41] [19]	eval-logloss:0.00106	train-logloss:0.00107
```

Once the training finishes, the model file should be written into
`/tmp/nvlfare/poc/site-1/${job_id}/test.model.json` and `/tmp/nvflare/poc/site-2/${job_id}/test.model.json`
respectively, where `job_id` is the UUID printed out when we ran `submit_job`.

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
