"""Shared defaults and constants for ARC-Challenge SFT."""

from pathlib import Path

DEFAULT_DATA_DIR = Path("/scratch/gautschi/fang375/verl_CL/data/commensense/ARC-Challenge")
DEFAULT_OUTPUT_DIR = Path("/scratch/gautschi/fang375/verl_CL/SFT_commensense_reasoning/outputs/arc_challenge_lora")
DEFAULT_MODEL = (
    "/scratch/gautschi/fang375/verl_CL/checkpoints/verl_grpo_example_test_CL_May21/"
    "test_cl_may21/global_step_435/actor_merged_hf"
)

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
