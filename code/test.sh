#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

DATAPATH="../user_data"

# 1. 为测试集图片池和 query 文本计算特征
split="test" 
resume="${DATAPATH}/experiments/vqa_finetune_vit-h-14_roberta-large_bs48_1gpu/checkpoints/epoch_latest.pt"
python -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 \
    --resume="${resume}" \
    --vision-model=ViT-H-14 \
    --text-model=RoBERTa-wwm-ext-large-chinese

# 2. 进行 KNN 检索，为测试集每个 query，匹配特征余弦相似度最高的 top-1 商品图片
python -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${split}_texts.txt_feat.jsonl" \
    --top-k=1 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${split}_predictions.jsonl"

# 3. 跨模态检索和视觉问答推理生成最终结果
python infer.py


