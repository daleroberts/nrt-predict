import subprocess
import textwrap
import joblib
import pytest
import signal
import socket
import time
import os

from nrtpredict import checksum_file

def test_serialise_model_pickle(tmp_path):
    from models import NoOp
    model_path = tmp_path / "model.pkl"

    model = NoOp()
    joblib.dump(model, model_path)

    with open(model_path, 'rb') as f:
        sha = checksum_file(f)

    of = tmp_path / "obs.tif"

    f = tmp_path / "test.yaml"
    f.write_text(textwrap.dedent(f"""\n
    models:
       - name: file://{model_path}:{sha}
         output: {of}
    """))

    subprocess.check_call(['./nrtpredict.py', '-c', f, 'data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])

    assert os.path.exists(of)
