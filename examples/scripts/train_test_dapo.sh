
export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0
export EXPERIMENT_PATH="/mnt/shared-storage-user/CFT"
export EXPERIMENT_NAME=$(date +%Y%m%d-%H:%M)
python3 -m CFT.cli.train_ppo_ray \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 4 \
   --vllm_gpu_memory_utilization 0.6 \
   --colocate_all_models \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --gamma 1.0 \
   --kl_estimator k1 \
   --advantage_estimator group_norm \
   --dynamic_filtering \
   --dynamic_filtering_reward_range -3 4\
   --eps_clip_low_high 0.2 0.3 \
   --pretrain ${EXPERIMENT_PATH}/CFT-main/CFT/datasets/model/Qwen3-4B-Thinking-2507\
   --remote_rm_url ${EXPERIMENT_PATH}/CFT-main/examples/python/rlla.py \
   --save_path ../test_scripts/final/${EXPERIMENT_NAME} \
   --ckpt_path ../test_scripts/ckpt/${EXPERIMENT_NAME} \
   --wandb_project CFT_train_CFT\
   --wandb_run_name ${EXPERIMENT_NAME} \
   --save_hf_ckpt \
   --micro_train_batch_size 64 \
   --train_batch_size 128 \
   --micro_rollout_batch_size  8\
   --rollout_batch_size 32 \
   --n_samples_per_prompt 8 \
   --use_dynamic_batch \
   --max_epochs 1 \
   --num_episodes 2 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --param_dtype bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data ${EXPERIMENT_PATH}/CFT-main/CFT/datasets/data/merged-train-filter.json \
   --input_key input \
   --label_key output \
   --sys_key instruction \
   --apply_chat_template \
   --gradient_checkpointing \
   --init_kl_coef -1\
   --packing_samples \
   --vllm_sync_backend nccl \
   --save_steps 20\
   --eval_dataset ${EXPERIMENT_PATH}/CFT-main/CFT/datasets/data/merged-eval-filter.json \
   --entropy_loss_coef 0 
   