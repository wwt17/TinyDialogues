#!/bin/bash

models=(
    ~/word-learning/ckpt/lm_dataset_name_childes_tokenizer_name_tokenizers\:childes_wordlevel_config_name_model_config\:pythia_12l_drop0.1.json_train_batch_size_32_learning_rate_0.001_weight_decay_0.3_no_weight_decay_embeddings_False_lr_scheduler_type_reduce_lr_on_plateau_seed_0/best/
    ~/word-learning/ckpt/lm_dataset_name_childes_tokenizer_name_tokenizers\:childes_wordlevel_config_name_model_config\:pythia_12l_drop0.1.json_train_batch_size_32_learning_rate_0.001_weight_decay_0.3_no_weight_decay_embeddings_False_lr_scheduler_type_reduce_lr_on_plateau_seed_1/best/
    ~/word-learning/ckpt/lm_dataset_name_childes_tokenizer_name_tokenizers\:childes_wordlevel_config_name_model_config\:pythia_12l_drop0.1.json_train_batch_size_32_learning_rate_0.001_weight_decay_0.3_no_weight_decay_embeddings_False_lr_scheduler_type_reduce_lr_on_plateau_seed_2/best/
    ~/word-learning/ckpt/lm_dataset_name_childes_tokenizer_name_tokenizers\:childes_wordlevel_config_name_model_config\:pythia_12l_drop0.1.json_train_batch_size_32_learning_rate_0.001_weight_decay_0.3_no_weight_decay_embeddings_False_lr_scheduler_type_reduce_lr_on_plateau_seed_3/best/
    ~/word-learning/ckpt/lm_dataset_name_childes_tokenizer_name_tokenizers\:childes_wordlevel_config_name_model_config\:pythia_12l_drop0.1.json_train_batch_size_32_learning_rate_0.001_weight_decay_0.3_no_weight_decay_embeddings_False_lr_scheduler_type_reduce_lr_on_plateau_seed_4/best/
)

for model in "${models[@]}"; do
    rm -r llm_devo_word_sim_results
    python eval_word_sim.py \
    --ckpt_path "${model}" \
    --tokenizer ~/word-learning/tokenizers/childes_wordlevel/
done
