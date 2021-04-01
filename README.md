# nrt-predict

Run prediction models on [Digital Earth Australia](https://www.ga.gov.au/dea) Near-Real-Time (NRT) satellite observations. The NRT data is an effort to acquire, atmospherically-correct, and package the data as quickly as possible from when a [Sentinel-2](http://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2) satellite flies over an area in Australia. **nrt-predict** provides a customisable workflow framework for running various machine learning, AI, and statistical models on the data to produce additional outputs. The aim is for the (user-defined) models to be both easy to write (minimal boilerplate) but also customisable (if needed). Ancillary datasets can be used by the models and **nrt-predict** retrieves and crops these datasets automatically for the model.

##

## Quickstart

### Install Dependencies (MacOS)
1. Docker - https://docs.docker.com/docker-for-mac/install/
1. GDAL: `brew install gdal`
1. Pipenv: `brew install pipenv`
1. PyEnv: `brew install pyenv`

### Configure The Environment
1. As a general rule, all configuration settings should be configured as environment variables. There is a template .env file with expected settings in `.env.template`
1. Copy the template - `cp .env.template .env`
1. Edit `.env` with your local environment variables. Note that `.env` is intentially excluded from the git report via `.gitignore`.
1. Load the environment variables in the current context - `source .env`

### Build/run locally
1. Initialise python 3.8 (version required defined within Pipfile) - `pyenv 3.8`
1. Install dependencies in virtual enviroment: `pipenv install`
1. Initialise virtual environment: `pipenv shell`
1. Run the NRT predict process: `python ./nrt_predict.py data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09`

### Alternative (if you don't want to use pipenv to manage your virutal environment)
1. Generate regular *requirements.txt*: `pipenv run pip freeze > requirements.txt`
1. (Optional) activate your environment (eg. `source ./venv/bin/activate` etc).
1. Install dependencies via pip: `pip install -r requirements.txt`
1. Run the NRT predict process: `python ./nrt_predict.py data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09`

### Build/Run the Docker container
1. `./run_docker_tests.sh`

### Run a model in Docker
1. `docker exec -it nft-predict bash`
1. Run the tests - `pytest ./tests`
1. Run a specific test model - `python ./nrt_predict.py data/test/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09`

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
