#!/usr/bin/env bash
# Resumable training: re-run this command until it prints "status": "done" (each call stops after MAX_MIN minutes).
# Usage: scripts/train_until_budget.sh <name> <run> <encoder> <fusion> [extra args]
set -e
NAME=$1; RUN=$2; ENC=$3; FUS=$4; shift 4
PY=${PYTHON:-python}
$PY -m acmta_sepsis.experiments train --name "$NAME" --run "$RUN" --encoder "$ENC" --fusion "$FUS" \
   --hidden ${HIDDEN:-64} --heads ${HEADS:-4} --dropout ${DROPOUT:-0.2} --lr ${LR:-1e-3} --weight-decay ${WD:-1e-4} \
   --batch-size ${BS:-256} --max-epochs ${MAXEP:-12} --patience ${PAT:-3} --seed ${SEED:-42} --threads ${THREADS:-2} \
   --max-minutes ${MAX_MIN:-9} "$@"
