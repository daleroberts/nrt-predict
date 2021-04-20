import subprocess
import textwrap
import pytest
import os

from osgeo import gdal

def test_empty_config_local(tmp_path):
    f = tmp_path / "test.yaml"
    f.touch()

    subprocess.check_call(['./nrtpredict.py', '-c', f, 'data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])

def test_cog_output(tmp_path):
    f = tmp_path / "test.yaml"
    of = tmp_path / "obs.tif"

    f.write_text(textwrap.dedent(f"""\n
    models:
       - name: NoOp
         output: {of}
         driver: COG
    """))

    subprocess.check_call(['./nrtpredict.py', '-c', f, 'data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])

    ds = gdal.Open(str(of))
    assert ds.GetMetadataItem("LAYOUT", "IMAGE_STRUCTURE") == "COG"
