import subprocess
import textwrap
import pytest
import signal
import socket
import time
import os

def test_program_in_cwd():
    assert os.path.exists("nrtpredict.py")

def test_help():
    status = os.system("./nrtpredict.py --help")
    assert status == 0
