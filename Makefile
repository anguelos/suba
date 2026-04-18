SPHINXBUILD  = sphinx-build
SPHINXOPTS   =
SOURCEDIR    = docs
BUILDDIR     = docs/_build

.PHONY: help docs test testcov clean

help:
	@echo "make docs     - build Sphinx HTML documentation"
	@echo "make test     - run all tests"
	@echo "make testcov  - run unit tests with coverage report"
	@echo "make clean    - remove build artefacts"

docs:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo "HTML docs written to $(BUILDDIR)/html/index.html"

test:
	python -m pytest tests/ -v

testcov:
	@mkdir -p tmp
	python -m pytest tests/unit/ -v --cov=suba --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:tmp/coverage.xml
	genbadge coverage -i tmp/coverage.xml -o tmp/coverage.svg
	@echo "Coverage HTML report written to htmlcov/index.html"
	@echo "Coverage badge written to tmp/coverage.svg"

clean:
	rm -rf $(BUILDDIR)
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf src/*.egg-info
	rm -rf dist/ build/
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.py[co]" -delete
	@echo "Clean done"
