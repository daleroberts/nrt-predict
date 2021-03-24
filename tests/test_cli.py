import subprocess
import pytest
import signal
import socket
import time
import os

PORT = 9000
HOST = 'localhost'
KEY = "testtesttest"

@pytest.fixture(scope="session")
def minio():
    # Setup
    
    os.environ['MINIO_ACCESS_KEY'] = KEY
    os.environ['MINIO_SECRET_KEY'] = KEY
    os.environ['MINT_MODE'] = 'full'
    os.environ['ACCESS_KEY'] = KEY
    os.environ['SECRET_KEY'] = KEY
    os.environ['ENABLE_HTTPS'] = "0"
    
    command = "minio server data"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, preexec_fn=os.setsid) 

    # wait for start
    
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

    # Teardown
    
    os.killpg(process.pid, signal.SIGTERM)

def write_gdalconfig_for_minio(f):
    f.write_text(f"""\n
    gdalconfig:
      GDAL_DISABLE_READDIR_ON_OPEN: YES
      CPL_VSIL_CURL_ALLOWED_EXTENSIONS: '.tif,.geojson'
      CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE: YES
      CPL_CURL_VERBOSE: YES
      CPL_DEBUG: YES
      AWS_HTTPS: NO
      AWS_VIRTUAL_HOSTING: FALSE
      AWS_S3_ENDPOINT: {HOST}:{PORT}
      AWS_SECRET_ACCESS_KEY: {KEY}
      AWS_ACCESS_KEY_ID: {KEY}
    """)

def test_gdal_with_minio(minio):
    from osgeo import gdal
    import numpy as np
    
    gdal.UseExceptions()
    gdal.SetConfigOption('AWS_HTTPS', 'NO')
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')
    gdal.SetConfigOption('AWS_VIRTUAL_HOSTING', 'FALSE')
    gdal.SetConfigOption('AWS_S3_ENDPOINT', f'{HOST}:{PORT}')
    gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', KEY)
    gdal.SetConfigOption('AWS_ACCESS_KEY_ID', KEY)
    
    path = '/vsis3/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09/NBAR/NBAR_B01.TIF'
    ds = gdal.Open(path)
    
    band = ds.GetRasterBand(1)
    
    xoff, yoff, xcount, ycount = (0, 0, 10, 10)
    data = band.ReadAsArray(xoff, yoff, xcount, ycount)
    assert np.sum(data) > 0

def test_empty_config_s3(minio, tmp_path):
    f = tmp_path / "test.yaml"
    write_gdalconfig_for_minio(f)
    subprocess.check_call(['./nrt_predict.py', '-c', f, 's3://test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])

def test_empty_config_local(minio, tmp_path):
    f = tmp_path / "test.yaml"
    subprocess.check_call(['./nrt_predict.py', '-c', f, 'data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])

def test_program_in_cwd():
    assert os.path.exists("nrt_predict.py")

def test_help():
    status = os.system("./nrt_predict.py --help")
    assert status == 0
