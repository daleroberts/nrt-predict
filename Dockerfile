FROM osgeo/gdal:ubuntu-small-3.2.2

# Copy the application to the container

COPY . /app
WORKDIR /app

# Update and install requirements

RUN apt-get -qq -y update && \
    apt-get install -qq --no-install-recommends python3-pip make && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip3 install -r requirements.txt

# Install minio for testing the code

ADD https://dl.min.io/server/minio/release/linux-amd64/minio minio
RUN chmod +x minio

