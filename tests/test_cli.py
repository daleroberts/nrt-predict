import subprocess
import textwrap
import pytest
import signal
import socket
import time
import os

PORT = 9000
HOST = os.environ['S3_HOST']
USER = os.environ['MINIO_ROOT_USERNAME']
PASSWORD = os.environ['MINIO_ROOT_PASSWORD']

@pytest.fixture(scope="session")
def minio():
    # Setup
    
    os.environ['MINIO_ACCESS_KEY'] = USER
    os.environ['MINIO_SECRET_KEY'] = PASSWORD
    os.environ['MINT_MODE'] = 'full'
    os.environ['ACCESS_KEY'] = USER
    os.environ['SECRET_KEY'] = PASSWORD
    os.environ['ENABLE_HTTPS'] = "0"
    
    # wait for Minio to become available - via Docker compose
    not_running = True
    while not_running:
        try:
            s = socket.socket()
            s.connect((HOST, PORT))
            not_running = False
        except Exception as e: 
            time.sleep(0.5)
        finally:
            s.close()
    
    yield

def write_gdalconfig_for_minio(f):
    f.write_text(textwrap.dedent(f"""\n
    gdalconfig:
      GDAL_DISABLE_READDIR_ON_OPEN: YES
      CPL_VSIL_CURL_ALLOWED_EXTENSIONS: '.tif,.geojson'
      CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE: YES
      #CPL_CURL_VERBOSE: YES
      CPL_DEBUG: YES
      AWS_HTTPS: NO
      AWS_VIRTUAL_HOSTING: FALSE
      AWS_S3_ENDPOINT: {HOST}:{PORT}
      AWS_SECRET_ACCESS_KEY: {PASSWORD}
      AWS_ACCESS_KEY_ID: {USER}
    """))

def test_gdal_with_minio(minio):
    from osgeo import gdal
    import numpy as np
    
    gdal.UseExceptions()
    gdal.SetConfigOption('AWS_HTTPS', 'NO')
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')
    gdal.SetConfigOption('AWS_VIRTUAL_HOSTING', 'FALSE')
    gdal.SetConfigOption('AWS_S3_ENDPOINT', f'{HOST}:{PORT}')
    gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', PASSWORD)
    gdal.SetConfigOption('AWS_ACCESS_KEY_ID', USER)
    
    path = './data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09/NBAR/NBAR_B01.TIF'
    ds = gdal.Open(path)
    
    band = ds.GetRasterBand(1)
    
    xoff, yoff, xcount, ycount = (0, 0, 10, 10)
    data = band.ReadAsArray(xoff, yoff, xcount, ycount)
    assert np.sum(data) > 0

def test_empty_config_s3(minio, tmp_path):
    f = tmp_path / "test.yaml"
    write_gdalconfig_for_minio(f)
    subprocess.check_call(['./nrt_predict.py', '-c', f, 's3://test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])

#def test_empty_config_http(minio, tmp_path):
#    f = tmp_path / "test.yaml"
#    write_gdalconfig_for_minio(f)
#    subprocess.check_call(['./nrt_predict.py', '-c', f, f'http://{KEY}:{KEY}@{HOST}:{PORT}/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])

def test_empty_config_local(minio, tmp_path):
    f = tmp_path / "test.yaml"
    subprocess.check_call(['./nrt_predict.py', '-c', f, './data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])

def test_program_in_cwd():
    assert os.path.exists("nrt_predict.py")

def test_ancillary_on_s3(minio, tmp_path):
    f = tmp_path / "test.yaml"
    g = tmp_path / "clip.geojson"
    f.write_text(textwrap.dedent("""
    clipshpfn: {g}
    models:
      - name: NoOp
        output: nbr.tif
        inputs:
          - filename: s3://minio/test/s2be.tif
    """))
    write_gdalconfig_for_minio(f)
    subprocess.check_call(['./nrt_predict.py', '-c', f, 's3://test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])
    #assert os.path.exists(g)

def test_help():
    status = os.system("./nrt_predict.py --help")
    assert status == 0
