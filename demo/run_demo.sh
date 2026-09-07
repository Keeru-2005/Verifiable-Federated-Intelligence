#!/usr/bin/env bash

# ==============================================================================
# VFI Phase 3 Review Demo Execution Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
node demo/run_demo.js "$@"
