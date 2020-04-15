import os
import tempfile
import unittest
import platform
import xgboost
import subprocess
import numpy


class TestCLI(unittest.TestCase):
    template = '''
booster = gbtree
objective = reg:squarederror
eta = 1.0
gamma = 1.0
seed = 0
min_child_weight = 0
max_depth = 3
task = {task}
model_in = {model_in}
model_out = {model_out}
test_path = {test_path}
name_pred = {name_pred}

num_round = 10
data = {data_path}
eval[test] = {data_path}
'''

    def test_cli_model(self):
        curdir = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
        project_root = os.path.normpath(
            os.path.join(curdir, os.path.pardir, os.path.pardir))
        data_path = "{root}/demo/data/agaricus.txt.train?format=libsvm".format(
            root=project_root)

        if platform.system() == 'Windows':
            exe = 'xgboost.exe'
        else:
            exe = 'xgboost'
        exe = os.path.join(project_root, exe)
        assert os.path.exists(exe)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_out = os.path.join(tmpdir, 'test_load_cli_model')
            config_path = os.path.join(tmpdir, 'test_load_cli_model.conf')

            train_conf = self.template.format(data_path=data_path,
                                              task='train',
                                              model_in='NULL',
                                              model_out=model_out,
                                              test_path='NULL',
                                              name_pred='NULL')
            with open(config_path, 'w') as fd:
                fd.write(train_conf)

            subprocess.run([exe, config_path])

            predict_out = os.path.join(tmpdir,
                                       'test_load_cli_model-prediction')
            predict_conf = self.template.format(task='pred',
                                                data_path=data_path,
                                                model_in=model_out,
                                                model_out='NULL',
                                                test_path=data_path,
                                                name_pred=predict_out)
            with open(config_path, 'w') as fd:
                fd.write(predict_conf)

            subprocess.run([exe, config_path])

            cli_predt = numpy.loadtxt(predict_out)

            parameters = {
                'booster': 'gbtree',
                'objective': 'reg:squarederror',
                'eta': 1.0,
                'gamma': 1.0,
                'seed': 0,
                'min_child_weight': 0,
                'max_depth': 3
            }
            data = xgboost.DMatrix(data_path)
            booster = xgboost.train(parameters, data, num_boost_round=10)
            py_predt = booster.predict(data)

            numpy.testing.assert_allclose(cli_predt, py_predt)

            cli_model = xgboost.Booster(model_file=model_out)
            cli_predt = cli_model.predict(data)
            numpy.testing.assert_allclose(cli_predt, py_predt)
