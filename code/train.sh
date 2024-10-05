#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export MKL_SERVICE_FORCE_INTEL=1

# Step1: 跨模态检索

GPUS_PER_NODE=2 # 卡数
WORKER_CNT=1 # 机器数
MASTER_ADDR="localhost"
MASTER_PORT=8514 # 同台机器同时起多个任务，请分别分配不同的端口号
RANK=0 

# 存放预训练参数和预处理好的数据集
DATAPATH="../user_data"

# 指定LMDB格式的训练集和验证集路径（存放了LMDB格式的图片和图文对数据）
train_data="${DATAPATH}/datasets/lmdb/train"
val_data="${DATAPATH}/datasets/lmdb/valid"
num_workers=4 # 训练集pytorch dataloader的进程数，设置为>0，以减小训练时读取数据的时间开销
valid_num_workers=4 # 验证集pytorch dataloader的进程数，设置为>0，以减小验证时读取数据的时间开销

# 指定刚刚下载好的Chinese-CLIP预训练权重的路径
resume="${DATAPATH}/pretrained_weights/clip_cn_vit-h-14.pt"
reset_data_offset="--reset-data-offset" # 从头读取训练数据
reset_optimizer="--reset-optimizer" # 重新初始化AdamW优化器

# 指定输出相关配置
output_base_dir="${DATAPATH}/experiments/"
name="vqa_finetune_vit-h-14_roberta-large_bs48_1gpu" # finetune超参、日志、ckpt将保存在../user_data/datasets/experiments/vqa_finetune_vit-h-14_roberta-base_bs48_1gpu/
save_step_frequency=999999 # disable it
save_epoch_frequency=1 # 每轮保存一个finetune ckpt
log_interval=10 # 日志打印间隔步数
report_training_batch_acc="--report-training-batch-acc" # 训练中，报告训练batch的in-batch准确率

# 指定训练超参数
context_length=52 # 序列长度，这里指定为Chinese-CLIP默认的52
warmup=100 # warmup步数
batch_size=48 # 训练单卡batch size
valid_batch_size=48 # 验证单卡batch size
lr=3e-6 # 学习率，因为这里我们使用的对比学习batch size很小，所以对应的学习率也调低一些
wd=0.001 # weight decay
max_epochs=5 # 训练轮数，也可通过--max-steps指定训练步数
valid_step_interval=1000 # 验证步数间隔
valid_epoch_interval=1 # 验证轮数间隔
vision_model="ViT-H-14" # 指定视觉侧结构为ViT-H/14
text_model="RoBERTa-wwm-ext-large-chinese" # 指定文本侧结构为RoBERTa-large
use_augment="--use-augment" # 对图像使用数据增强
grad_checkpointing="--grad-checkpointing" # 激活重计算策略，用更多训练时间换取更小的显存开销

torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
      --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} cn_clip/training/main.py \
      --train-data=${train_data} \
      --val-data=${val_data} \
      --num-workers=${num_workers} \
      --valid-num-workers=${valid_num_workers} \
      --resume=${resume} \
      ${reset_data_offset} \
      ${reset_optimizer} \
      --logs-specs=${output_base_dir} \
      --name=${name} \
      --save-step-frequency=${save_step_frequency} \
      --save-epoch-frequency=${save_epoch_frequency} \
      --log-interval=${log_interval} \
      ${report_training_batch_acc} \
      --context-length=${context_length} \
      --warmup=${warmup} \
      --batch-size=${batch_size} \
      --valid-batch-size=${valid_batch_size} \
      --valid-step-interval=${valid_step_interval} \
      --valid-epoch-interval=${valid_epoch_interval} \
      --lr=${lr} \
      --wd=${wd} \
      --max-epochs=${max_epochs} \
      --vision-model=${vision_model} \
      ${use_augment} \
      ${grad_checkpointing} \
      --text-model=${text_model}


# Step2: 视觉问答

cd swift
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 swift sft \
  --model_type minicpm-v-v2_6-chat \
  --model_id_or_path OpenBMB/MiniCPM-V-2_6 \
  --sft_type lora \
  --num_train_epochs 3 \
  --learning_rate 0.0005 \
  --eval_steps 2000 \
  --save_steps 1000 \
  --output_dir ../../user_data/experiments \
  --dataset ../../user_data/datasets/qa_pairs/train.jsonl \
  --val_dataset ../../user_data/datasets/qa_pairs/val.jsonl\
  --deepspeed default-zero2

