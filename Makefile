.PHONY: install test clean

install:
	python -m pip install -e .

test:
	python -m unittest discover tests

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
