# 大模型图文问答挑战赛

科大讯飞xDatawhale [大模型图文问答挑战赛](https://challenge.xfyun.cn/topic/info?type=graphic-quiz-challenge&option=ssgy)第二名方案分享，包含预训练权重的代码可从[百度网盘](https://pan.baidu.com/s/1ESEVM66a3rmcMg7KS5nfsA?pwd=kww5)下载。

## 解决方案及算法介绍

赛题要求能够准确地从图像和相关的文本描述中提取信息，并回答关于这些信息的问题，测试集提问存在以下两种类型：

* 类型 1：通过提问（question）对提问图片（related_image）进行提问和描述
* 类型 2：通过提问（question）检索到最相关的图片（related_image）

针对类型 1，将其转换为**视觉问答**任务，从训练集标注中提取文本信息，设计 QA 对生成视觉问答数据集，详见 [generate_qa.py](code/generate_qa.py)。使用开源框架 [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) 的 [MiniCPM-V 2.6](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6) 模型对数据集 Lora 微调。

针对类型 2，将其转换为**跨模态检索**任务，数据集构建详见 [transform.py](code/transform.py)。基于开源框架 [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) 的 [clip_cn_vit-h-14.pt](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-h-14.pt) 模型在对构造的数据集微调。

## 安装要求

运行下列命令即可安装本项目所需的三方库：

```bash
cd code
conda env create -f environment.yml
conda activate llm-qa

cd swift
pip install -e .[llm]
cd ..
```

部分依赖如下：

* Ubuntu 18.04.5 LTS
* Python 3.10.14
* pytorch==2.3.1
* CUDA Version: 12.0

## 数据集预处理

### 跨模态检索

生成跨模态检索数据集，将赛题 `train_annotation.csv` 按照 8:2 比例划分为训练集和验证集，`query.json` 作为测试集。

```bash
# 1. 将赛题原始数据拷贝到指定目录便于预处理
mkdir -p ../user_data/datasets && cp -r ../xfdata/* ../user_data/datasets
# 2. 生成 ${split}_texts.jsonl，并将图片以 base64 形式分别存放在 ${split}_imgs.tsv 文件中
python transform.py
# 3. 将 tsv 和 jsonl 文件一起序列化，转换为内存索引的 LMDB 数据库文件
python cn_clip/preprocess/build_lmdb_dataset.py --data_dir ../user_data/datasets --splits train,valid,test
```

### 视觉问答

设计 QA 对，生成视觉问答数据集。

```bash
python generate_qa.py
```

## 训练

下载预训练模型，并执行训练脚本：

```bash
# clip_cn_vit-h-14.pt
wget https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-h-14.pt
mkdir -p ../user_data/pretrained_weights && mv clip_cn_vit-h-14.pt ../user_data/pretrained_weights
# 跨模态检索 & 视觉问答
bash train.sh
```

跨模态检索训练完成的模型存储在 `../user_data/experiments/vqa_finetune_vit-h-14_roberta-large_bs48_1gpu/checkpoints/epoch_latest.pt`。

视觉问答训练完成的模型存储在 `../user_data/experiments/minicpm-v-v2_6-chat/v8-20240912-123632/checkpoint-3000` 目录。

## 预测

运行下列命令得到最终结果：

```bash
bash test.sh
```

最终结果存储在 `../prediction_result/result.json`

## 引用

感谢以下开源项目的帮助：

* https://github.com/OFA-Sys/Chinese-CLIP
* https://github.com/modelscope/ms-swift