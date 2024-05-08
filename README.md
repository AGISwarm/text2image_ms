# Python Template Project

This repository provides a Python project template optimized for development, testing, and CI/CD integration using Docker.

## Project Structure

- `src/AGISwarm/python_template/`: Main package.
  - `__init__.py`: Package initialization.
  - `__main__.py`: Main module logic.
  - `py.typed`: Marker file for PEP 561 typing.
- `pyproject.toml`: Specifies build system requirements and project dependencies.
- `tests/`: Unit and functional tests.
- `.github/workflows/`: 
  - `build-analysis-test.yml`: CI for build and test.
  - `build-pypi-docker.yml`: CI for Docker and PyPI deployment.
- `dockerfile`: Docker setup for Python.

## Features

- Automated tests with examples.
- CI/CD via GitHub Actions.
- Docker integration for deployment.

## Getting Started

### Installation

1. Import the template repository to your GitHub account.

2. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
3. Navigate to the project directory:
   ```bash
   cd <repository-name>
   ```
4. Adapt the project to your needs:
   1. Modify the project name in `pyproject.toml` and `src/AGISwarm/python_template/__init__.py` to match your project name.
   2. Change namespace `AGISwarm` to your own namespace.
   3. Add your code to `src/<namespace>/<project-name>/` folder.
   4. Write your tests in `tests/` folder.
      1. Unittests will be run automatically by GitHub Actions every time you push to the repository. Functional tests will be not.
   
5. Install dependencies:
   ```bash
   pip install .
   ```

### CICD Setup

For testing and deployment in a CI/CD setup, refer to our [CICD project](https://github.com/AGISwarm/CICD).

### Usage

#### Command Line

```bash
python -m <namespace>.<project-name>
```

#### Docker

1. Build python package:
   ```bash
   python -m build
   ```
2. Create PIP_INDEX_EXTRA_URL environment variable to be able to install dependencies from a self-hosted PyPI server.

   Given URL is self-hosted PyPI server from [CICD](README.md#cicd-setup) setup. 
   Replace with your own if needed. 
   Or remove secret mounting in Dockerfile, if no need. 
   ```bash
   export PIP_INDEX_EXTRA_URL=http://pypi-server/
   ```
3. Build Docker image. Use this command for testing purposes only. If you use [CICD](README.md#cicd-setup) setup, it will build and push the image for you after creating a tag in your repository:
   ```bash
   docker build -t <your tag name> --secret id=PIP_INDEX_EXTRA_URL,env=PIP_INDEX_EXTRA_URL .
   ```
4. Run Docker container:
   ```bash
   docker run <your tag name>
   ```


## License

Distributed under the MIT License. See `LICENSE.txt` for details.

## Contact

Denis Diachkov - diachkov.da@gmail.com