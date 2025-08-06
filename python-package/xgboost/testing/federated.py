# pylint: disable=unbalanced-tuple-unpacking, too-many-locals
"""Tests for federated learning."""

import multiprocessing
import os
import subprocess
import tempfile
import time
from typing import List, cast

from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.model_selection import train_test_split

import xgboost as xgb
import xgboost.federated
from xgboost import testing as tm

from .._typing import EvalsLog
from ..collective import _Args as CollArgs

SERVER_KEY = "server-key.pem"
SERVER_CERT = "server-cert.pem"
CLIENT_KEY = "client-key.pem"
CLIENT_CERT = "client-cert.pem"


def run_server(port: int, world_size: int, with_ssl: bool) -> None:
    """Run federated server for test."""
    if with_ssl:
        xgboost.federated.run_federated_server(
            world_size,
            port,
            server_key_path=SERVER_KEY,
            server_cert_path=SERVER_CERT,
            client_cert_path=CLIENT_CERT,
        )
    else:
        xgboost.federated.run_federated_server(world_size, port)


def run_worker(
    port: int, world_size: int, rank: int, with_ssl: bool, device: str
) -> None:
    """Run federated client worker for test."""
    comm_env: CollArgs = {
        "dmlc_communicator": "federated",
        "federated_server_address": f"localhost:{port}",
        "federated_world_size": world_size,
        "federated_rank": rank,
    }
    if with_ssl:
        comm_env["federated_server_cert_path"] = SERVER_CERT
        comm_env["federated_client_key_path"] = CLIENT_KEY
        comm_env["federated_client_cert_path"] = CLIENT_CERT

    cpu_count = os.cpu_count()
    assert cpu_count is not None
    n_threads = cpu_count // world_size

    # Always call this before using distributed module
    with xgb.collective.CommunicatorContext(**comm_env):
        # Load file, file will not be sharded in federated mode.
        X, y = load_svmlight_file(f"agaricus.txt-{rank}.train")
        dtrain = xgb.DMatrix(X, y)
        X, y = load_svmlight_file(f"agaricus.txt-{rank}.test")
        dtest = xgb.DMatrix(X, y)

        # Specify parameters via map, definition are same as c++ version
        param = {
            "max_depth": 2,
            "eta": 1,
            "objective": "binary:logistic",
            "nthread": n_threads,
            "tree_method": "hist",
            "device": device,
        }

        # Specify validations set to watch performance
        watchlist = [(dtest, "eval"), (dtrain, "train")]
        num_round = 20

        # Run training, all the features in training API is available.
        results: EvalsLog = {}
        bst = xgb.train(
            param,
            dtrain,
            num_round,
            evals=watchlist,
            early_stopping_rounds=2,
            evals_result=results,
        )
        assert tm.non_increasing(cast(List[float], results["train"]["logloss"]))
        assert tm.non_increasing(cast(List[float], results["eval"]["logloss"]))

        # save the model, only ask process 0 to save the model.
        if xgb.collective.get_rank() == 0:
            with tempfile.TemporaryDirectory() as tmpdir:
                bst.save_model(os.path.join(tmpdir, "model.json"))
            xgb.collective.communicator_print("Finished training\n")


def run_federated(world_size: int, with_ssl: bool, use_gpu: bool) -> None:
    """Launcher for clients and the server."""
    port = 9091

    server = multiprocessing.Process(
        target=run_server, args=(port, world_size, with_ssl)
    )
    server.start()
    time.sleep(1)
    if not server.is_alive():
        raise ValueError("Error starting Federated Learning server")

    workers = []
    for rank in range(world_size):
        device = f"cuda:{rank}" if use_gpu else "cpu"
        worker = multiprocessing.Process(
            target=run_worker, args=(port, world_size, rank, with_ssl, device)
        )
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
    server.terminate()


def run_federated_learning(with_ssl: bool, use_gpu: bool, test_path: str) -> None:
    """Run federated learning tests."""
    n_workers = 2

    if with_ssl:
        command = "openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout {part}-key.pem -out {part}-cert.pem -subj /C=US/CN=localhost"  # pylint: disable=line-too-long
        server_key = command.format(part="server").split()
        subprocess.check_call(server_key)
        client_key = command.format(part="client").split()
        subprocess.check_call(client_key)

    train_path = os.path.join(tm.data_dir(test_path), "agaricus.txt.train")
    test_path = os.path.join(tm.data_dir(test_path), "agaricus.txt.test")

    X_train, y_train = load_svmlight_file(train_path)
    X_test, y_test = load_svmlight_file(test_path)

    X0, X1, y0, y1 = train_test_split(X_train, y_train, test_size=0.5)
    X0_valid, X1_valid, y0_valid, y1_valid = train_test_split(
        X_test, y_test, test_size=0.5
    )

    dump_svmlight_file(X0, y0, "agaricus.txt-0.train")
    dump_svmlight_file(X0_valid, y0_valid, "agaricus.txt-0.test")

    dump_svmlight_file(X1, y1, "agaricus.txt-1.train")
    dump_svmlight_file(X1_valid, y1_valid, "agaricus.txt-1.test")

    run_federated(world_size=n_workers, with_ssl=with_ssl, use_gpu=use_gpu)
