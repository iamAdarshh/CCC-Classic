#!/usr/bin/env bash
# Run the level 6 beam search on all scored inputs, sharded across CPU cores, then merge.
#
# Usage: ./run_all.sh [width] [topk] [shards_per_file] [config]
#   width            beam width (default 64; 128 is slower but a bit better)
#   topk             moves expanded per beam state (default 8)
#   shards_per_file  parallel processes per input file (default: cores / 3)
#   config           index into beam.CONFIGS (default 0)
#
# Run from the level-6 directory. Results only ever replace a program with a shorter verified one.
set -euo pipefail
cd "$(dirname "$0")"

W=${1:-64}
K=${2:-8}
CORES=$(python3 -c "import os; print(os.cpu_count())")
N=${3:-$(( CORES / 3 > 0 ? CORES / 3 : 1 ))}
C=${4:-0}
PY=${PY:-python3}
OUT=shards
mkdir -p "$OUT"

FILES="2-medium 3-large 4-enormous"
echo "width=$W topk=$K shards/file=$N config=$C python=$PY"

for f in $FILES; do
  for ((i = 0; i < N; i++)); do
    "$PY" shard.py "in_level-6_$f.txt" "$OUT/$f.w$W.c$C.$i.json" "$i" "$N" "$W" "$K" "$C" \
      > "$OUT/$f.w$W.c$C.$i.log" 2>&1 &
  done
done

wait
for f in $FILES; do
  "$PY" merge.py "in_level-6_$f.txt" "out_level-6_$f.txt" "$OUT"/$f.w$W.c$C.*.json
done
