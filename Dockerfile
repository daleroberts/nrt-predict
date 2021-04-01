FROM osgeo/gdal:ubuntu-small-3.2.2

RUN apt-get -qq update
RUN apt-get install -qq python3-pip
RUN apt-get install wget
RUN pip3 install pipenv

COPY . ./
# Pipenv Settings
RUN export LC_ALL='en_US.UTF-8'
RUN export LANG='en_US.UTF-8'
RUN pipenv install --sequential --system







