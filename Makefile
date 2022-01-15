.PHONY: venv
venv:
	pip install -U pip setuptools wheel pipenv
	pipenv install -e . --dev

.PHONY: install
install:
	pipenv install .

.PHONY: test
test:
	pipenv run pytest

.PHONY: lint
lint:
	pipenv run flake8

.PHONY: fix
fix:
	pipenv run black -l79 ./ineqpy/

