import subprocess
import textwrap
import pytest
import signal
import socket
import time
import os

def test_empty_config_local(tmp_path):
    f = tmp_path / "test.yaml"
    subprocess.check_call(['./nrtpredict.py', '-c', f, 'data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])
