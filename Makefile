install:
	conda install torch torchvision
	pip install tensorboard
	pip install -e .

test:
	python -m unittest discover -s rlil -p "*test.py"

autopep8:
	autopep8 --in-place --recursive . 

tensorboard:
	tensorboard --logdir runs

benchmark:
	tensorboard --logdir benchmarks/runs --port=6007

clean:
	rm -rf dist
	rm -rf build

build: clean
	python setup.py sdist bdist_wheel

deploy: lint test build
	twine upload dist/*
