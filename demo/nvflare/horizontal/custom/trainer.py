import os

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

import xgboost as xgb
from xgboost import callback


class SupportedTasks(object):
    TRAIN = "train"


class XGBoostTrainer(Executor):
    def __init__(self, server_address: str, world_size: int, server_cert_path: str,
                 client_key_path: str, client_cert_path: str, use_gpus: bool):
        """Trainer for federated XGBoost.

        Args:
            server_address: address for the gRPC server to connect to.
            world_size: the number of sites.
            server_cert_path: the path to the server certificate file.
            client_key_path: the path to the client key file.
            client_cert_path: the path to the client certificate file.
        """
        super().__init__()
        self._server_address = server_address
        self._world_size = world_size
        self._server_cert_path = server_cert_path
        self._client_key_path = client_key_path
        self._client_cert_path = client_cert_path
        self._use_gpus = use_gpus

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext,
                abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Executing {task_name}")
        try:
            if task_name == SupportedTasks.TRAIN:
                self._do_training(fl_ctx)
                return make_reply(ReturnCode.OK)
            else:
                self.log_error(fl_ctx, f"{task_name} is not a supported task.")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except BaseException as e:
            self.log_exception(fl_ctx,
                               f"Task {task_name} failed. Exception: {e.__str__()}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _do_training(self, fl_ctx: FLContext):
        client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        rank = int(client_name.split('-')[1]) - 1
        communicator_env = {
            'xgboost_communicator': 'federated',
            'federated_server_address': self._server_address,
            'federated_world_size': self._world_size,
            'federated_rank': rank,
            'federated_server_cert': self._server_cert_path,
            'federated_client_key': self._client_key_path,
            'federated_client_cert': self._client_cert_path
        }
        with xgb.collective.CommunicatorContext(**communicator_env):
            # Load file, file will not be sharded in federated mode.
            dtrain = xgb.DMatrix('agaricus.txt.train?format=libsvm')
            dtest = xgb.DMatrix('agaricus.txt.test?format=libsvm')

            # Specify parameters via map, definition are same as c++ version
            param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
            if self._use_gpus:
                self.log_info(fl_ctx, f'Training with GPU {rank}')
                param['tree_method'] = 'gpu_hist'
                param['gpu_id'] = rank

            # Specify validations set to watch performance
            watchlist = [(dtest, 'eval'), (dtrain, 'train')]
            num_round = 20

            # Run training, all the features in training API is available.
            bst = xgb.train(param, dtrain, num_round, evals=watchlist,
                            early_stopping_rounds=2, verbose_eval=False,
                            callbacks=[callback.EvaluationMonitor(rank=rank)])

            # Save the model.
            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
            run_dir = workspace.get_run_dir(run_number)
            bst.save_model(os.path.join(run_dir, "test.model.json"))
            xgb.collective.communicator_print("Finished training\n")
