# nrt-predict
Predict from NRT data

## Quickstart

### Install Dependencies (MacOS)
1. Docker - https://docs.docker.com/docker-for-mac/install/
1. GDAL: `brew install gdal`
1. Pipenv: `brew install pipenv`

### Configure Environment
1. `cp .env.template .env`
1. Edit `.env` with your local environment variables. Note that `.env` is intentially excluded from the git report via `.gitignore`.
1. `source .env`

### Build/run locally
1. `pipenv install`
1. `pipenv shell`
1. `python ./nrt_predict.py s3://dea-public-data/L2/sentinel-2-nrt/S2MSIARD/2021-03-25/S2A_OPER_MSI_ARD_TL_VGS4_20210325T014951_A030057_T56HKH_N02.09`

### Build/Run the Docker container
1. `./run_in_docker.sh`

### Run a model in Docker
1. `docker exec -it nft-predict bash`
1. `python ./nrt_predict.py s3://dea-public-data/L2/sentinel-2-nrt/S2MSIARD/2021-03-25/S2A_OPER_MSI_ARD_TL_VGS4_20210325T014951_A030057_T56HKH_N02.09`
