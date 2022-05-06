# Federated XGBoost Demo

This directory contains a demo of Federated Learning using [NVFlare](https://nvidia.github.io/).

To run the demo, first install NVFlare:
```console
pip install nvflare
```

Prepare the data:
```console
./prepare_data.sh
```

Start the NVFlare federated server:
```console
./poc/server/startup/start.sh
```

In another terminal, start the first worker:
```console
./poc/site-1/startup/start.sh
```

And the second worker:
```console
./poc/site-2/startup/start.sh
```

Then start the admin CLI, using `admin/admin` as username/password:
```console
./poc/admin/startup/fl_admin.sh
```

In the admin CLI, run the following commands:
```console
upload_app hello-xgboost
set_run_number 1
deploy_app hello-xgboost all
start_app all
```

Once the training finishes, the model file should be written into
`./poc/site-1/run_1/test.model.json` and `./poc/site-2/run_1/test.model.json`
respectively.

Finally, shutdown everything from the admin CLI:
```console
shutdown client
shutdown server
```
