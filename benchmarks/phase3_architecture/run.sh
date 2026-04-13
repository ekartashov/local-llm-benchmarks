#!/usr/bin/env bash
# Phase 3 thin wrapper — delegates to run_phase3_sequence.sh
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/run_phase3_sequence.sh" "$@"
