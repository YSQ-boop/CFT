
export WITHLENGTH=0
export REFINEDREWARD=0
export INTERMEDIATEREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0
export MODEL_PATH="Qwen3-4B-Thinking-2507"
export EXPERIMENT_PATH=""
export EXPERIMENT_NAME=$(date +%Y%m%d-%H:%M)

python3 -m CFT.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.6 \
   --colocate_all_models \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --gamma 1.0 \
   --kl_estimator k1 \
   --advantage_estimator rloo \
   --pretrain ${EXPERIMENT_PATH}/CFT-main/CFT/datasets/model/Qwen3-4B-Thinking-2507 \
   --remote_rm_url ${EXPERIMENT_PATH}/CFT-main/examples/python/CFT.py \
   --save_path ../test_scripts/final/${EXPERIMENT_NAME} \
   --ckpt_path ../test_scripts/ckpt/${EXPERIMENT_NAME} \
   --wandb_project CFT_train\
   --wandb_run_name ${EXPERIMENT_NAME} \
   --save_hf_ckpt \
   --micro_train_batch_size 64 \
   --train_batch_size 256 \
   --micro_rollout_batch_size 8 \
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
   --vllm_sync_backend nccl \
   --save_steps 30\
   --init_kl_coef -1 \
   --sample_kl_mode topk \
   --sample_kl_topk 20 \
   --sample_kl_topk_add_tail \
   --policy_loss_type ppo \
   --enable_adv_kl_reweight\
   --adv_kl_weight_clip 0.2\
   --policy_loss_student_entropy_top_ratio 1 \
   --policy_loss_kl_top_ratio 0 \
   --policy_loss_entropy_diff_top_ratio 0.2 \
   --policy_loss_teacher_entropy_top_ratio 0 \
   --entropy_loss_coef 0 \
   --eps_clip_low_high 0.2 0.3 \
   --load_checkpoint \
   --max_ckpt_num 1 
   
