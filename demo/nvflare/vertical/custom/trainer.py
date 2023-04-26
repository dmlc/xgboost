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
                 client_key_path: str, client_cert_path: str):
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
            if rank == 0:
                label = '&label_column=0'
            else:
                label = ''
            dtrain = xgb.DMatrix(f'higgs.train.csv?format=csv{label}', data_split_mode=1)
            dtest = xgb.DMatrix(f'higgs.test.csv?format=csv{label}', data_split_mode=1)

            # specify parameters via map
            param = {
                'validate_parameters': True,
                'eta': 0.1,
                'gamma': 1.0,
                'max_depth': 8,
                'min_child_weight': 100,
                'tree_method': 'approx',
                'grow_policy': 'depthwise',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
            }

            # specify validations set to watch performance
            watchlist = [(dtest, "eval"), (dtrain, "train")]
            # number of boosting rounds
            num_round = 10

            bst = xgb.train(param, dtrain, num_round, evals=watchlist, early_stopping_rounds=2)

            # Save the model.
            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
            run_dir = workspace.get_run_dir(run_number)
            bst.save_model(os.path.join(run_dir, "higgs.model.federated.vertical.json"))
            xgb.collective.communicator_print("Finished training\n")
