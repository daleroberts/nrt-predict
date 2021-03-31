# nrt-predict

Run prediction models on [Digital Earth Australia](https://www.ga.gov.au/dea) Near-Real-Time (NRT) satellite observations. The NRT data is an effort to acquire, atmospherically-correct, and package the data as quickly as possible from when a [Sentinel-2](http://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2) satellite flies over an area in Australia. **nrt-predict** provides a customisable workflow framework for running various machine learning, AI, and statistical models on the data to produce additional outputs. The aim is for the (user-defined) models to be both easy to write (minimal boilerplate) but also customisable (if needed). Ancillary datasets can be used by the models and **nrt-predict** retrieves and crops these datasets automatically for the model.

##

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
1. Install dependencies in virtual enviroment: `pipenv install`
1. Initialise virtual environment: `pipenv shell`
1. Run the NRT predict process: `python ./nrt_predict.py s3://dea-public-data/L2/sentinel-2-nrt/S2MSIARD/2021-03-25/S2A_OPER_MSI_ARD_TL_VGS4_20210325T014951_A030057_T56HKH_N02.09`

### Alternative (if you don't want to use pipenv to manage your virutal environment)
1. Generate regular *requirements.txt*: `pipenv run pip freeze > requirements.txt`
1. (Optional) activate your environment (eg. `source ./venv/bin/activate` etc).
1. Install dependencies via pip: `pip install -r requirements.txt`
1. Run the NRT predict process: `python ./nrt_predict.py s3://dea-public-data/L2/sentinel-2-nrt/S2MSIARD/2021-03-25/S2A_OPER_MSI_ARD_TL_VGS4_20210325T014951_A030057_T56HKH_N02.09`

### Build/Run the Docker container
1. `./run_in_docker.sh`

### Run a model in Docker
1. `docker exec -it nft-predict bash`
1. `python ./nrt_predict.py s3://dea-public-data/L2/sentinel-2-nrt/S2MSIARD/2021-03-25/S2A_OPER_MSI_ARD_TL_VGS4_20210325T014951_A030057_T56HKH_N02.09`

## Information

### NRT Data

The Digital Earth Australia NRT data can be found in Amazon s3 buckets of the form:
```
s3://dea-public-data/L2/sentinel-2-nrt/S2MSIARD/<date>/<package>
```
The data can also be accessed through HTTPS at:
```
https://data.dea.ga.gov.au/L2/sentinel-2-nrt/S2MSIARD/<date>/<package>
```

A (minified!) version of a package layout can be found in the [data/test](https://github.com/daleroberts/nrt-predict/tree/main/data/test/) directory of this repo.
