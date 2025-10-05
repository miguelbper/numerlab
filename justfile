# List all available recipes
default:
    just --list

# Check that all programs are installed
[group("installation")]
check-versions:
    uv --version  # https://docs.astral.sh/uv/
    just --version  # https://github.com/casey/just
    direnv --version  # https://direnv.net/

# Allow direnv to load environment variables
[group("installation")]
direnv-allow:
    direnv allow

# Create uv virtual environment
[group("installation")]
create-venv:
    uv sync

# Install pre-commit hooks
[group("installation")]
install-pre-commit:
    uv run pre-commit install

# Setup MLFlow (reminder)
[group("installation")]
reminder-mlflow:
    @echo "\033[1;33mRemember to setup MLFlow!\033[0m"

# Setup environment variables (reminder)
[group("installation")]
reminder-env-vars:
    @echo "\033[1;33mRemember to setup the environment variables by editing the .envrc file!\033[0m"

# Setup repo
[group("installation")]
setup: direnv-allow create-venv install-pre-commit reminder-mlflow reminder-env-vars

# Run pre-commit hooks
[group("linting & formatting")]
pre-commit:
    uv run pre-commit run --all

# Run tests
[group("testing")]
test:
    uv run pytest

# Run tests with coverage
[group("testing")]
test-cov:
    uv run pytest --cov=src --cov-report=html

# Print tree of the project (requires installing tree)
[group("tools")]
tree:
    tree -a -I ".venv|.git|.pytest_cache|.coverage|dist|__pycache__|.vscode|.ruff_cache" --dirsfirst

# Run mlflow server
[group("tools")]
mlflow-server:
    uv run mlflow server --backend-store-uri logs/mlflow/mlruns

# Clean logs directory
[group("cleanup")]
clean-logs:
    rm -rf logs/* && touch logs/.gitkeep

# Clean data directory
[group("cleanup")]
clean-data:
    rm -rf data/* && touch data/.gitkeep

# Download dataset
[group("numerai")]
download-dataset:
    uv run src/numerai/data/download_dataset.py
