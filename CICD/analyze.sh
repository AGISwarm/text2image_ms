python -m pylint src
python -m pyright src
python -m black src --check
python -m isort src --check-only