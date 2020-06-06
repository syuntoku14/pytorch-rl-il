install:
	conda install torch torchvision
	pip install tensorboard
	pip install -e .

test:
	pytest -v --benchmark-skip

benchmark:
	pytest -v --benchmark-only

autopep8:
	autopep8 --in-place --recursive . 

tensorboard:
	tensorboard --logdir runs

clean:
	rm -rf dist
	rm -rf build
	find . -type f -name '*.so' -delete
	find . -type f -name '*.cpp' -delete

build:
	python setup.py build_ext --inplace 

deploy: lint test build
	twine upload dist/*
