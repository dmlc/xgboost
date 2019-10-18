import time

from xgboost import RabitTracker
import xgboost as xgb


def test_rabit_tracker():
    tracker = RabitTracker(hostIP='127.0.0.1', nslave=1)
    tracker.start(1)
    rabit_env = [
        str.encode('DMLC_TRACKER_URI=127.0.0.1'),
        str.encode('DMLC_TRACKER_PORT=9091'),
        str.encode('DMLC_TASK_ID=0')]
    xgb.rabit.init(rabit_env)
    ret = xgb.rabit.broadcast('test1234', 0)
    assert str(ret) == 'test1234'
    xgb.rabit.finalize()
