.PHONY: install
install:
	pipenv install .

.PHONY: test
test:
	pipenv run pytest

.PHONY: lint
lint:
	pipenv run flake8 ./ineqpy

.PHONY: fix
fix:
	pipenv run black -l79 ./ineqpy/

.PHONY: venv
venv:
	pip install -U pip setuptools wheel pipenv
	pipenv --python 3
	pipenv install --dev
