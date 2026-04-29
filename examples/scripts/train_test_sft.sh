set -x
export EXPERIMENT_PATH=""
export EXPERIMENT_NAME=$(date +%Y%m%d-%H:%M)
read -r -d '' training_commands <<EOF
CFT.cli.train_sft \
   --max_len 8192 \
   --dataset ${EXPERIMENT_PATH}/CFT-main/CFT/datasets/data/merged-train-filter.json \
   --input_key input \
   --output_key output \
   --apply_chat_template \
   --train_batch_size 256 \
   --micro_train_batch_size 16 \
   --max_samples 500000 \
   --pretrain ${EXPERIMENT_PATH}/CFT-main/CFT/datasets/model/Qwen3-4B-Thinking-2507 \
   --save_path ../test_scripts/final/${EXPERIMENT_NAME} \
   --save_steps 5 \
   --logging_steps 1 \
   --eval_steps -1 \
   --save_hf_ckpt \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi