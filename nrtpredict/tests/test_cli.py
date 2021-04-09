import subprocess
import textwrap
import pytest
import signal
import socket
import time
import os

AWS_S3_HOSTNAME = "localhost"
AWS_S3_PORT = 9000
AWS_S3_ENDPOINT = f'http://localhost:9000'
AWS_ACCESS_KEY_ID = 'minio'
AWS_SECRET_ACCESS_KEY = 'minio123'

NRT_PREDICT_BINARY_PATH = r"./nrt_predict.py"

@pytest.fixture(scope="session")
def minio():
    # Setup
    os.environ['ENABLE_HTTPS'] = "0"
    
    command = "minio server ./data"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, preexec_fn=os.setsid) 

    # wait for start
    
    not_running = True

    try:
        s = socket.socket()
        s.connect((AWS_S3_HOSTNAME, AWS_S3_PORT))
        not_running = False
    except Exception as e: 
        time.sleep(0.1)
    finally:
        s.close()
    
    yield

    # Teardown
    
    os.killpg(process.pid, signal.SIGTERM)

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
      AWS_S3_ENDPOINT: {AWS_S3_ENDPOINT}
      AWS_SECRET_ACCESS_KEY: {AWS_SECRET_ACCESS_KEY}
      AWS_ACCESS_KEY_ID: {AWS_ACCESS_KEY_ID}
    """))

# TODO: Is this actually using minio?
def test_gdal_with_minio(minio):
    from osgeo import gdal
    import numpy as np
    
    gdal.UseExceptions()
    gdal.SetConfigOption('AWS_HTTPS', 'NO')
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')
    gdal.SetConfigOption('AWS_VIRTUAL_HOSTING', 'FALSE')
    gdal.SetConfigOption('AWS_S3_ENDPOINT', f'{AWS_S3_ENDPOINT}')
    gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', AWS_SECRET_ACCESS_KEY)
    gdal.SetConfigOption('AWS_ACCESS_KEY_ID', AWS_ACCESS_KEY_ID)

    
    path = './data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09/NBAR/NBAR_B01.TIF'
    ds = gdal.Open(path)
    
    band = ds.GetRasterBand(1)
    
    xoff, yoff, xcount, ycount = (0, 0, 10, 10)
    data = band.ReadAsArray(xoff, yoff, xcount, ycount)
    assert np.sum(data) > 0

def test_empty_config_s3(minio, tmp_path):
    f = tmp_path / "test.yaml"
    write_gdalconfig_for_minio(f)
    subprocess.check_call([NRT_PREDICT_BINARY_PATH, '-c', f, 's3://test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])

#def test_empty_config_http(minio, tmp_path):
#    f = tmp_path / "test.yaml"
#    write_gdalconfig_for_minio(f)
#    subprocess.check_call(['../nrt_predict.py', '-c', f, f'http://{KEY}:{KEY}@{HOST}:{PORT}/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])

def test_empty_config_local(minio, tmp_path):
    f = tmp_path / "test.yaml"
    subprocess.check_call([NRT_PREDICT_BINARY_PATH, '-c', f, './data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])

def test_program_in_cwd():
    assert os.path.exists(NRT_PREDICT_BINARY_PATH)

def test_ancillary_on_s3(minio, tmp_path):
    f = tmp_path / "test.yaml"
    g = tmp_path / "clip.geojson"
    f.write_text(textwrap.dedent("""
    clipshpfn: {g}
    models:
      - name: NoOp
        output: nbr.tif
        inputs:
          - filename: s3://test/s2be.tif
    """))
    write_gdalconfig_for_minio(f)
    subprocess.check_call([NRT_PREDICT_BINARY_PATH, '-c', f, 's3://test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09'])
    #assert os.path.exists(g)

def test_help():
    status = os.system(f'{NRT_PREDICT_BINARY_PATH} --help')
    assert status == 0
