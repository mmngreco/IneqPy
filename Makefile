.PHONY: pipenv
pipenv:
	pip install -U pip setuptools wheel pipenv --user

.PHONY: dev
dev:
	pipenv install -e . --dev

.PHONY: venv
venv: pipenv dev

.PHONY: install
install:
	pipenv install .

.PHONY: test
test:
	pipenv run pytest

.PHONY: lint
lint:
	pipenv run flake8 src

.PHONY: fix
fix:
	pipenv run black -l79 ./ineqpy/

.PHONY: doc
doc:
	pipenv run mkdocs build

.PHONY: doc-serve
doc-serve:
	pipenv run mkdocs build

.PHONY: vim
vim:
	pipenv run nvim .

.PHONY: build
build:
	pipenv run python setup.py sdist bdist_wheel
