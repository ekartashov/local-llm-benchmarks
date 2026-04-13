#!/usr/bin/env bash
# Phase 1 entry point — delegates to run_phase1_sequence.sh.
# Called by: just phase1
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${REPO_ROOT}/benchmarks/phase1_engine_selection/run_phase1_sequence.sh" "$@"
