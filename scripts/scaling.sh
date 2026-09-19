#!/usr/bin/env bash
# Reproduce the README tables: bandwidth ceiling + every backend at three grid sizes.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${BIN:-build/heat}
mkdir -p results docs
$BIN --bandwidth | tee docs/bandwidth.txt
$BIN --all --n 512  --steps 8000 --threads 1,2,4,8,10 --csv results/scaling-n512.csv  | tee docs/scaling-n512.txt
$BIN --all --n 2000 --steps 500  --threads 1,2,4,8,10 --csv results/scaling-n2000.csv --pgm docs/heat-2000.pgm | tee docs/scaling-n2000.txt
$BIN --all --n 4000 --steps 200  --threads 1,4,8      --csv results/scaling-n4000.csv | tee docs/scaling-n4000.txt
