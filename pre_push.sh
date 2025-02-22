#!/bin/sh

# Get the absolute path to the Git repository
REPO_ROOT=$(git rev-parse --show-toplevel)

# Run pre-commit hooks with the specified Python interpreter
python -m pre_commit run --config "$REPO_ROOT/.pre-push-config.yaml" --all-files --hook-stage push