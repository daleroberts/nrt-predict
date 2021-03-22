import pytest
import os

def test_program_in_cwd():
    assert os.path.exists("nrt_predict.py")

def test_help():
    status = os.system("./nrt_predict.py --help")
    assert status == 0

def test_import():
    import nrt_predict
    nrt_predict.main(obstmp='/tmp', url='S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09')
