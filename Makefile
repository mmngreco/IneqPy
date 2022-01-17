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
	pipenv run flake8 src

.PHONY: fix
fix:
	pipenv run black -l79 ./ineqpy/


.PHONY: doc
doc:
	cd docs && pipenv run make html && xdg-open build/html/index.html

.PHONY: vim
vim:
	pipenv run nvim .
