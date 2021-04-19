define HELP
make test|doc|upload|clean

  deps        - install dependencies
  test        - run tests
  doc         - build documentation
  upload      - upload to pypi
  clean       - clean repo
endef
export HELP

help:
	@echo "$$HELP"

deps:
	python -m pip install -r requirements.txt 

test: 
	pytest

build-docker: Dockerfile requirements.txt
	#docker kill nrt-predict
	docker container rm -f nrt-predict
	docker build . -t nrt-predict 

test-docker: build-docker
	docker run --name nrt-predict -it nrt-predict:latest pytest

clean:
	@rm -fr data/S2*
	@rm -fr *.tif
	@rm -fr *.json
	@rm -fr tests/__pycache__
	@rm -fr models/__pycache__

doc: 
	git add README.md
	git commit -m 'Update README'
	git push

upload:
	python3 setup.py sdist
	twine upload dist/*
	

.PHONY: clean test doc upload deps build-docker test-docker
