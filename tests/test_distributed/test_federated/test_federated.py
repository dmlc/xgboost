#!/usr/bin/python
import multiprocessing
import sys
import time

import xgboost as xgb
import xgboost.federated

SERVER_KEY = 'server-key.pem'
SERVER_CERT = 'server-cert.pem'
CLIENT_KEY = 'client-key.pem'
CLIENT_CERT = 'client-cert.pem'


def run_server(port: int, world_size: int, with_ssl: bool) -> None:
    if with_ssl:
        xgboost.federated.run_federated_server(port, world_size, SERVER_KEY, SERVER_CERT,
                                               CLIENT_CERT)
    else:
        xgboost.federated.run_federated_server(port, world_size)


def run_worker(port: int, world_size: int, rank: int, with_ssl: bool, with_gpu: bool) -> None:
    communicator_env = {
        'xgboost_communicator': 'federated',
        'federated_server_address': f'localhost:{port}',
        'federated_world_size': world_size,
        'federated_rank': rank
    }
    if with_ssl:
        communicator_env['federated_server_cert'] = SERVER_CERT
        communicator_env['federated_client_key'] = CLIENT_KEY
        communicator_env['federated_client_cert'] = CLIENT_CERT

    # Always call this before using distributed module
    with xgb.collective.CommunicatorContext(**communicator_env):
        # Load file, file will not be sharded in federated mode.
        dtrain = xgb.DMatrix('agaricus.txt.train-%02d' % rank)
        dtest = xgb.DMatrix('agaricus.txt.test-%02d' % rank)

        # Specify parameters via map, definition are same as c++ version
        param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
        if with_gpu:
            param['tree_method'] = 'gpu_hist'
            param['gpu_id'] = rank

        # Specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 20

        # Run training, all the features in training API is available.
        bst = xgb.train(param, dtrain, num_round, evals=watchlist,
                        early_stopping_rounds=2)

        # Save the model, only ask process 0 to save the model.
        if xgb.collective.get_rank() == 0:
            bst.save_model("test.model.json")
            xgb.collective.communicator_print("Finished training\n")


def run_federated(with_ssl: bool = True, with_gpu: bool = False) -> None:
    port = 9091
    world_size = int(sys.argv[1])

    server = multiprocessing.Process(target=run_server, args=(port, world_size, with_ssl))
    server.start()
    time.sleep(1)
    if not server.is_alive():
        raise Exception("Error starting Federated Learning server")

    workers = []
    for rank in range(world_size):
        worker = multiprocessing.Process(target=run_worker,
                                         args=(port, world_size, rank, with_ssl, with_gpu))
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
    server.terminate()


if __name__ == '__main__':
    run_federated(with_ssl=True, with_gpu=False)
    run_federated(with_ssl=False, with_gpu=False)
    run_federated(with_ssl=True, with_gpu=True)
    run_federated(with_ssl=False, with_gpu=True)
