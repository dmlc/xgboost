import os
import tempfile
import unittest
import platform
import xgboost
import subprocess
import numpy
import json


class TestCLI(unittest.TestCase):
    template = '''
booster = gbtree
objective = reg:squarederror
eta = 1.0
gamma = 1.0
seed = {seed}
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

    curdir = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
    project_root = os.path.normpath(
        os.path.join(curdir, os.path.pardir, os.path.pardir))

    def get_exe(self):
        if platform.system() == 'Windows':
            exe = 'xgboost.exe'
        else:
            exe = 'xgboost'
        exe = os.path.join(self.project_root, exe)
        assert os.path.exists(exe)
        return exe

    def test_cli_model(self):
        data_path = "{root}/demo/data/agaricus.txt.train?format=libsvm".format(
            root=self.project_root)
        exe = self.get_exe()
        seed = 1994

        with tempfile.TemporaryDirectory() as tmpdir:
            model_out_cli = os.path.join(tmpdir, 'test_load_cli_model-cli.bin')
            model_out_py = os.path.join(tmpdir, 'test_cli_model-py.bin')
            config_path = os.path.join(tmpdir, 'test_load_cli_model.conf')

            train_conf = self.template.format(data_path=data_path,
                                              seed=seed,
                                              task='train',
                                              model_in='NULL',
                                              model_out=model_out_cli,
                                              test_path='NULL',
                                              name_pred='NULL')
            with open(config_path, 'w') as fd:
                fd.write(train_conf)

            subprocess.run([exe, config_path])

            predict_out = os.path.join(tmpdir,
                                       'test_load_cli_model-prediction')
            predict_conf = self.template.format(task='pred',
                                                seed=seed,
                                                data_path=data_path,
                                                model_in=model_out_cli,
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
                'seed': seed,
                'min_child_weight': 0,
                'max_depth': 3
            }
            data = xgboost.DMatrix(data_path)
            booster = xgboost.train(parameters, data, num_boost_round=10)
            booster.save_model(model_out_py)
            py_predt = booster.predict(data)

            numpy.testing.assert_allclose(cli_predt, py_predt)

            cli_model = xgboost.Booster(model_file=model_out_cli)
            cli_predt = cli_model.predict(data)
            numpy.testing.assert_allclose(cli_predt, py_predt)

            with open(model_out_cli, 'rb') as fd:
                cli_model_bin = fd.read()
            with open(model_out_py, 'rb') as fd:
                py_model_bin = fd.read()

            assert hash(cli_model_bin) == hash(py_model_bin)

    def test_cli_help(self):
        exe = self.get_exe()
        completed = subprocess.run([exe], stdout=subprocess.PIPE)
        error_msg = completed.stdout.decode('utf-8')
        ret = completed.returncode
        assert ret == 1
        assert error_msg.find('Usage') != -1
        assert error_msg.find('eval[NAME]') != -1

        completed = subprocess.run([exe, '-V'], stdout=subprocess.PIPE)
        msg = completed.stdout.decode('utf-8')
        assert msg.find('XGBoost') != -1
        v = xgboost.__version__
        if v.find('SNAPSHOT') != -1:
            assert msg.split(':')[1].strip() == v.split('-')[0]
        else:
            assert msg.split(':')[1].strip() == v

    def test_cli_model_json(self):
        exe = self.get_exe()
        data_path = "{root}/demo/data/agaricus.txt.train?format=libsvm".format(
            root=self.project_root)
        seed = 1994

        with tempfile.TemporaryDirectory() as tmpdir:
            model_out_cli = os.path.join(
                tmpdir, 'test_load_cli_model-cli.json')
            config_path = os.path.join(tmpdir, 'test_load_cli_model.conf')

            train_conf = self.template.format(data_path=data_path,
                                              seed=seed,
                                              task='train',
                                              model_in='NULL',
                                              model_out=model_out_cli,
                                              test_path='NULL',
                                              name_pred='NULL')
            with open(config_path, 'w') as fd:
                fd.write(train_conf)

            subprocess.run([exe, config_path])
            with open(model_out_cli, 'r') as fd:
                model = json.load(fd)

            assert model['learner']['gradient_booster']['name'] == 'gbtree'
