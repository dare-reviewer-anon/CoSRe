#!/usr/bin/env bash
set -euo pipefail

# Activate env (adjust if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mini

TASK="InstallingAPrinter"
MAX_STEPS=1000
TOTAL_TIMESTEPS=5000000
PARTIAL_OBS=True

# Maximum number of trainings to run at once
MAX_PARALLEL=4

LOG_ROOT="logs"
mkdir -p "${LOG_ROOT}"

# Timestamp ID generator
gen_id() {
    date +"%Y%m%d_%H%M%S"
}

ROOM_SIZES=(7 8 9 10)
running_jobs=0

for ROOM_SIZE in "${ROOM_SIZES[@]}"; do
  ID=$(gen_id)
  LOGDIR="${LOG_ROOT}/${TASK}_room${ROOM_SIZE}_${ID}"
  mkdir -p "${LOGDIR}"

  LOGFILE="${LOGDIR}/train.log"

  echo ">>> Launching ${TASK}, room_size=${ROOM_SIZE}, ID=${ID}"
  echo "    Log: ${LOGFILE}"

  # Each job: its own nohup + its own log
  nohup python -u train_rl_agent.py \
      --task "${TASK}" \
      --room_size "${ROOM_SIZE}" \
      --max_steps "${MAX_STEPS}" \
      --total_timesteps "${TOTAL_TIMESTEPS}" \
      --partial_obs "${PARTIAL_OBS}" \
      > "${LOGFILE}" 2>&1 &

  pid=$!
  echo "    PID: ${pid}"

  running_jobs=$((running_jobs + 1))

  # Limit number of parallel jobs
  if (( running_jobs >= MAX_PARALLEL )); then
    wait -n    # wait for one job to finish
    running_jobs=$((running_jobs - 1))
  fi
done

wait   # wait for all background jobs to complete
echo "All room sizes finished."
