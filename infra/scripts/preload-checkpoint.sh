#!/usr/bin/env bash
# Usage: preload-checkpoint.sh <checkpoint_path>
# Issues posix_fadvise(POSIX_FADV_WILLNEED) to pre-warm checkpoint file(s) into OS page cache.
# Returns immediately (async) — the kernel loads pages in the background.
set -euo pipefail

TARGET="${1:-}"
if [ -z "${TARGET}" ]; then
  echo "Usage: $0 <path_to_file_or_directory>" >&2
  exit 1
fi
if [ ! -e "${TARGET}" ]; then
  echo "ERROR: Path not found: ${TARGET}" >&2
  exit 1
fi

START_MS=$(date +%s%3N)

python3 - "${TARGET}" << 'PYEOF'
import os, sys

def preload_file(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        size = os.fstat(fd).st_size
        if size > 0:
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_WILLNEED)
        os.close(fd)
        return size
    except Exception as e:
        print(f"[preload] Warning: Failed to preload {path}: {e}")
        return 0

target = sys.argv[1]
total_bytes = 0
file_count = 0

if os.path.isdir(target):
    for root, dirs, files in os.walk(target):
        for f in files:
            path = os.path.join(root, f)
            if os.path.isfile(path):
                total_bytes += preload_file(path)
                file_count += 1
else:
    total_bytes += preload_file(target)
    file_count += 1

print(f"[preload] WILLNEED issued for {file_count} files ({total_bytes/1e9:.2f} GB total)")
PYEOF

END_MS=$(date +%s%3N)
echo "[preload] Completed in $(( END_MS - START_MS )) ms (kernel loading pages async)"
