#!/usr/bin/env bash
# Phase 2 entry point — delegates to run_phase2_sequence.sh.
# Called by: just phase2
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${REPO_ROOT}/benchmarks/phase2_model_selection/run_phase2_sequence.sh" "$@"
