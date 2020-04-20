import os
import subprocess
import sys
import pytest


CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEMO_DIR = os.path.join(ROOT_DIR, 'demo', 'guide-python')


def test_basic_walkthrough():
    script = os.path.join(DEMO_DIR, 'basic_walkthrough.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)
    os.remove('dump.nice.txt')
    os.remove('dump.raw.txt')


def test_custom_multiclass_objective():
    script = os.path.join(DEMO_DIR, 'custom_softmax.py')
    cmd = ['python', script, '--plot=0']
    subprocess.check_call(cmd)


def test_custom_rmsle_objective():
    major, minor = sys.version_info[:2]
    if minor < 6:
        pytest.skip('Skipping RMLSE test due to Python version being too low.')
    script = os.path.join(DEMO_DIR, 'custom_rmsle.py')
    cmd = ['python', script, '--plot=0']
    subprocess.check_call(cmd)
