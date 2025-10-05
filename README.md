<div align="center">

# numerlab
[![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![PyTorch Lightning](https://img.shields.io/badge/-Lightning-7e4fff?logo=lightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Configs-Hydra-89b8cd)](https://hydra.cc/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) <br>
[![Code Quality](https://github.com/miguelbper/lightning-hydra-zen-template/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/miguelbper/lightning-hydra-zen-template/actions/workflows/code-quality.yaml)
[![Unit Tests](https://github.com/miguelbper/lightning-hydra-zen-template/actions/workflows/tests.yaml/badge.svg)](https://github.com/miguelbper/lightning-hydra-zen-template/actions/workflows/tests.yaml)

My codebase for the Numerai machine learning tournament

</div>

---

## Description
An ML codebase for the [Numerai Tournament](https://docs.numer.ai/). Allows for training PyTorch and sklearn/XGBoost/LightGBM models under a single interface.

Tools:
- **[uv](https://docs.astral.sh/uv/)**: Fast Python package installer and resolver
- **[just](https://github.com/casey/just)**: Simple command runner for project tasks
- **[direnv](https://direnv.net/)**: Auto-loads environment variables from `.envrc`
- **[pre-commit](https://pre-commit.com/)** and **[ruff](https://docs.astral.sh/ruff/)**: Code quality, linting, and formatting
- **[pytest](https://docs.pytest.org/)**: Tests with coverage configuration
- **[Hydra](https://hydra.cc/)** and **[hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/)**: Typed, composable experiment configs
- **[Lightning](https://lightning.ai/docs/pytorch/stable/)**: Organized PyTorch code
- **[MLflow](https://mlflow.org/)**: Experiment tracking (local server helper)


## Project Structure
```shell
.
├── .github                       # GitHub workflows and CI configuration
├── data                          # Downloaded datasets (gitignored)
├── logs                          # MLflow runs, logs, and artifacts
├── notebooks                     # Jupyter notebooks for exploration
├── scripts                       # Utility and maintenance scripts
├── src                           # Source code
│   └── numerai                   # Main package
│       ├── configs               # Hydra config schemas and groups
│       │   ├── experiments       # Experiment presets
│       │   ├── groups            # Config groups for components
│       │   │   ├── __init__.py
│       │   │   ├── callbacks.py
│       │   │   ├── data.py
│       │   │   ├── debug.py
│       │   │   ├── hydraconf.py
│       │   │   ├── logger.py
│       │   │   ├── model.py
│       │   │   ├── module.py
│       │   │   └── trainer.py
│       │   ├── utils             # Config utilities (paths/helpers)
│       │   │   ├── __init__.py
│       │   │   ├── paths.py
│       │   │   └── utils.py
│       │   ├── __init__.py
│       │   ├── deploy.py         # Deploy configuration entry
│       │   └── train.py          # Train configuration entry
│       ├── data                  # Data loading and preparation utilities
│       │   ├── __init__.py
│       │   ├── datamodule.py     # Dataset abstraction and splits
│       │   └── download.py       # Download Numerai datasets
│       ├── features              # Feature engineering and selection
│       ├── funcs                 # High-level train/deploy routines
│       │   ├── __init__.py
│       │   ├── deploy.py         # Build/upload predict function
│       │   └── train.py          # Training orchestration function
│       ├── metrics               # Metrics computation utilities
│       ├── model                 # Model definitions and architectures
│       ├── sklearn_              # sklearn-compatible wrappers/adapters
│       ├── utils                 # Logging, typing, and general helpers
│       ├── __init__.py
│       ├── deploy.py             # Script entrypoint for deployment
│       └── train.py              # Script entrypoint for training
├── tests                         # Test suite
├── .envrc                        # Environment variables (used by direnv)
├── .gitignore
├── .pre-commit-config.yaml       # Pre-commit hooks configuration
├── .python-version               # Pinned Python version for the project
├── README.md
├── justfile                      # Command runner with common tasks
├── pyproject.toml                # Project configuration and dependencies
└── uv.lock                       # Locked dependency versions
```

## Installation
Install the suggested tooling: [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just), and [direnv](https://direnv.net/). See each page for OS-specific install steps.

Then, in the project root:

```shell
# Allow direnv to load environment variables (create .envrc first if needed)
direnv allow

# Create virtual environment and install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

Environment variables needed for deployment (NumerAPI credentials) can be set in `.envrc` and auto-loaded by direnv:

```shell
export NUMERAI_PUBLIC_ID=...
export NUMERAI_SECRET_KEY=...
```

## Usage
Common workflows:

- Download the latest Numerai dataset locally:
```shell
uv run src/numerai/data/download.py
```

- Run tests and coverage:
```shell
just test
just test-cov
```

- Start a local MLflow tracking server:
```shell
just mlflow-server
```

- Train a model:
```shell
uv run src/numerai/train.py <options>
```

- Train and deploy (to Numerai Model Uploads) a model:
```shell
uv run src/numerai/deploy.py <options>
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Some helpful links about good practices and tooling for ML:
- https://cookiecutter-data-science.drivendata.org/opinions/
- https://rdrn.me/postmodern-python/
