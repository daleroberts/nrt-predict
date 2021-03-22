import pytest
import os

def test_program_in_cwd():
    assert os.path.exists("nrt_predict.py")

def test_help():
    status = os.system("./nrt_predict.py --help")
    assert status == 0
