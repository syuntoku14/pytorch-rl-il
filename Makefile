install:
	conda install torch torchvision
	pip install tensorboard
	pip install -e .

lint:
	pylint rlil --rcfile=.pylintrc

test:
	python -m unittest discover -s rlil -p "*test.py"

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
