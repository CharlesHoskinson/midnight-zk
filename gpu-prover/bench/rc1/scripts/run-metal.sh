#!/usr/bin/env bash
set -euo pipefail
# Include the load gate, key loading, verification and publication in the time limit.
# macOS has no `timeout`; use GNU coreutils' `gtimeout` when it is installed.
limit=(timeout --kill-after=3s 690s)
command -v timeout >/dev/null || { command -v gtimeout >/dev/null && limit=(gtimeout --kill-after=3s 690s) || limit=(); }
exec "${limit[@]}" bash gpu-prover/bench/rc1/scripts/run-set.sh metal "$@"
