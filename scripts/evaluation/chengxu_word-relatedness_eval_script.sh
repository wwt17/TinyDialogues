#!/bin/bash

# Shared project root and evaluation entrypoint for all runs.
PROJECT_ROOT="/mnt/d/babyLM_project"
EVAL_SCRIPT="eval_word_sim.py"

# Run word-relatedness evaluation for every model in a set.
# Args:
#   $1:  directory containing the model checkpoints
#   $2:  results text file (appended across models)
#   $3:  best-layer scores CSV (one row per model)
#   $4+: model names to evaluate
run_eval_set() {
    local ckpt_dir="$1"
    local output_file="$2"
    local output_csv="$3"
    shift 3

    for model_name in "$@"; do
        python "${EVAL_SCRIPT}" --ckpt_path "${ckpt_dir}/${model_name}" \
            --output_file "${output_file}" \
            --output_csv "${output_csv}"

        if [ $? -eq 0 ]; then
            echo "Successfully processed ${model_name}."
        else
            echo "Error processing ${model_name}."
        fi
    done
}

# TinyDialogues models.
tinydialogue_dir="${PROJECT_ROOT}/tinydialogue"
tinydialogue_models=(
    "GPT2-small_tinydialogue_ordered_10n_1e-04"
    "GPT2-small_tinydialogue_reversed_10n_1e-04"
    "GPT2-small_tinydialogue_randomized_10n_1e-04"
)
run_eval_set \
    "${tinydialogue_dir}/trained_GPT2_models" \
    "${tinydialogue_dir}/chengxu_word-relatedness_GPT2-small_tinydialogue_all_experiments_results.txt" \
    "${tinydialogue_dir}/chengxu_word-relatedness_GPT2-small_tinydialogue_all_experiments_best-layer-scores.csv" \
    "${tinydialogue_models[@]}"

# CHILDES models.
childes_dir="${PROJECT_ROOT}/CHILDES"
childes_models=(
    "Buckets/GPT2-small_CHILDES_ordered_5b_10n_1e-04"
    "Buckets/Randomized/GPT2-small_CHILDES_randomized_5b_10n_1e-04"
    "Buckets/Reversed/GPT2-small_CHILDES_reversed_5b_10n_1e-04"
)
run_eval_set \
    "${childes_dir}/trained_GPT2_models" \
    "${childes_dir}/chengxu_word-relatedness_GPT2-small_CHILDES_all_experiments_results.txt" \
    "${childes_dir}/chengxu_word-relatedness_GPT2-small_CHILDES_all_experiments_best-layer-scores.csv" \
    "${childes_models[@]}"