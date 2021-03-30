define HELP
make test|doc|upload|clean

  test        - run tests
  doc         - build documentation
  upload      - upload to pypi
  clean       - clean repo
endef
export HELP

help:
	@echo "$$HELP"

test:
	pytest

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
	

.PHONY: clean test doc upload
