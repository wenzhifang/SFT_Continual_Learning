#!/bin/bash
#SBATCH -A cgb
#SBATCH -q preemptible
#SBATCH -p ai
#SBATCH --gpus-per-node=1
#SBATCH -n14
#SBATCH -N1
#SBATCH -J sft_arc_challenge_lora
#SBATCH -o /scratch/gautschi/fang375/verl_CL/SFT_commensense_reasoning/sft_arc_challenge_lora_%j.out
#SBATCH -e /scratch/gautschi/fang375/verl_CL/SFT_commensense_reasoning/sft_arc_challenge_lora_%j.err
#SBATCH --time=24:00:00

set -xeuo pipefail

module load conda/2024.09
conda activate /home/fang375/.conda/envs/llama_slora

WORKDIR="/scratch/gautschi/fang375/verl_CL/SFT_commensense_reasoning"
cd "${WORKDIR}"

DATA_DIR=${DATA_DIR:-/scratch/gautschi/fang375/verl_CL/data/commensense/ARC-Challenge}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/gautschi/fang375/verl_CL/SFT_commensense_reasoning/outputs/arc_challenge_lora}

MODEL_PATH="/scratch/gautschi/fang375/verl_CL/checkpoints/verl_grpo_example_test_CL_May21/test_cl_may21/global_step_435/actor_merged_hf"

EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-4}
SEED=${SEED:-42}

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Error: Missing merged HF checkpoint: ${MODEL_PATH}/config.json"
    exit 1
fi

if [ ! -f "${DATA_DIR}/train.json" ]; then
    echo "Error: Missing training file: ${DATA_DIR}/train.json"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python main.py \
    --model-path "${MODEL_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --save-every-epoch

echo "LoRA adapter saved to ${OUTPUT_DIR}"
