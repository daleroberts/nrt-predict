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

RUN wget --quiet --output-document ./minio https://dl.min.io/server/minio/release/linux-amd64/minio
RUN chmod +x ./minio

# Include location of minio binary in $PATH
RUN echo "export PATH=/:${PATH}" >> /root/.bashrc







