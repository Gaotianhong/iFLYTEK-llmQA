import os
import re
import json
import torch
from tqdm import tqdm
from swift.utils import seed_everything
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)
from peft import PeftModel


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

DATAPATH = "../user_data"
# 为测试集图片池和query文本计算特征
split = "test"
query_path = f"{DATAPATH}/datasets/query.json"
test_texts_path = f"{DATAPATH}/datasets/test_texts.jsonl"
output_path = f"{DATAPATH}/datasets/{split}_predictions.jsonl"

# 视觉问答
model_type = ModelType.minicpm_v_v2_6_chat
model_id_or_path = None
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')
model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, model_id_or_path=model_id_or_path,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
# Load the LoRA model
lora_path = "../user_data/experiments/minicpm-v-v2_6-chat/v8-20240912-123632/checkpoint-3000"
model = PeftModel.from_pretrained(model, lora_path).eval()
seed_everything(42)


def match_first_text(query_text, test_texts):
    for test in test_texts:
        if query_text == test['text']:
            return test['text_id']
    return None


test_texts = []
with open(test_texts_path, 'r') as f:
    for line in f:
        test_texts.append(json.loads(line))

# Read query.json
with open(query_path, 'r') as f:
    query_data = json.load(f)

match_number = 0
for query in tqdm(query_data, desc="Processing queries"):
    match = re.search(r'请匹配到与(.+?)最相关的图片', query["question"])

    if match:  # 跨模态检索
        query_text = match.group(1).strip()
        text_id = match_first_text(query_text, test_texts)  # Find the first matching text_id
        match_number += 1
        # print(text_id)
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if record["text_id"] == text_id:
                    target_image_id = record["image_ids"][0]
                    query["answer"] = target_image_id
                    break
    else:  # 视觉问答
        query_input = "<image>" + query['question']
        images = [os.path.join('../user_data/datasets/image', query['related_image'])]
        response, history = inference(model, template, query_input, images=images)
        print(f"query: {query['question']} response: {response}")

        query["answer"] = response


print(f'matched queries={match_number}')

result_path = f"../prediction_result/result.json"
with open(result_path, 'w', encoding='utf-8') as json_file:
    json.dump(query_data, json_file, ensure_ascii=False, indent=4)
