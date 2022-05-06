import os

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

import xgboost as xgb


class SupportedTasks(object):
    TRAIN = "train"


class XGBoostTrainer(Executor):
    def __init__(self, server_address: str, world_size: int, server_cert_path: str,
                 client_key_path: str, client_cert_path: str):
        """Trainer for federated XGBoost.

        Args:
            data_root: directory with local train/test data.
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
        rabit_env = [
            f'federated_server_address={self._server_address}',
            f'federated_world_size={self._world_size}',
            f'federated_rank={rank}',
            f'federated_server_cert={self._server_cert_path}',
            f'federated_client_key={self._client_key_path}',
            f'federated_client_cert={self._client_cert_path}'
        ]
        xgb.rabit.init([e.encode() for e in rabit_env])

        # Load file, file will not be sharded in federated mode.
        dtrain = xgb.DMatrix('agaricus.txt.train-%s' % client_name)
        dtest = xgb.DMatrix('agaricus.txt.test-%s' % client_name)

        # Specify parameters via map, definition are same as c++ version
        param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}

        # Specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 20

        # Run training, all the features in training API is available.
        bst = xgb.train(param, dtrain, num_round, evals=watchlist,
                        early_stopping_rounds=2)

        # Save the model.
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = workspace.get_run_dir(run_number)
        bst.save_model(os.path.join(run_dir, "test.model.json"))
        xgb.rabit.tracker_print("Finished training\n")

        # Notify the tracker all training has been successful
        # This is only needed in distributed training.
        xgb.rabit.finalize()
