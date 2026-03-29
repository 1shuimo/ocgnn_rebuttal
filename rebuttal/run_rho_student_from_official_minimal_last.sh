#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NUM_EPOCH="${NUM_EPOCH:-1500}"
USE_NORMREG="${USE_NORMREG:-1}"

for ds in Amazon reddit tolokers photo YelpChi-all; do
  python rebuttal/rho_labeledNormal_our.py \
    --dataset "$ds" \
    --teacher_path "../rho_teacher_best_pth/${ds}_rho_teacher_official_minimal_last.pth" \
    --use_normreg "${USE_NORMREG}" \
    --num_epoch "${NUM_EPOCH}"
done
