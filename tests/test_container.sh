#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="llmzip:test"

echo "=== Building Docker image '${IMAGE_NAME}' ==="
docker build -t "${IMAGE_NAME}" .

echo "=== Testing container binary execution ==="
docker run --rm "${IMAGE_NAME}" --version
docker run --rm "${IMAGE_NAME}" --help

echo "=== Container verification passed successfully! ==="
