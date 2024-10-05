import os
import re
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def find_clothing_type(text):
    clothing_types = ['短袖', '长袖', '衬衫', '连衣裙', '裤子', '牛仔裤', '运动鞋', '风衣', '大衣', 'T恤', '裙子',
                      '外套', '羽绒服', '马甲', '西装', '背心', '短裙', '长裙', '半身裙', '长裤', '短裤', '上衣',
                      '毛衣', '卫衣', '针织衫', '背带裙', '休闲裤', '连体裤', '打底裤', '阔腿裤', '西服', '衬衫裙',
                      '睡衣', '家居服', '内衣', '泳衣', '比基尼', '吊带', '吊带裙', '吊带裤', '吊带背心', '吊带连衣裙',
                      '棉袄',]

    # 提取实体中的衣服类型关键词
    def extract_clothing_type(text, clothing_types):
        # 将衣服类型列表转换为正则表达式模式
        clothing_pattern = '|'.join(clothing_types)
        match = re.search(clothing_pattern, text)
        if match:
            return match.group()
        return None

    # 从识别结果中提取衣服类型
    clothing_type = extract_clothing_type(text, clothing_types)
    return clothing_type


def find_chinese_color(text):
    # 扩展的中文颜色列表，包含不同深浅和类型的颜色
    colors = [
        '红色', '蓝色', '绿色', '黄色', '黑色', '白色', '灰色', '橙色', '紫色', '粉色', '棕色',
        '金色', '银色', '青色', '靛色', '浅蓝色', '深蓝色', '亮黄色', '橘黄色', '深红色', '浅绿色'
    ]

    # 将颜色列表转换为正则表达式模式
    color_pattern = '|'.join(colors)

    # 使用正则表达式匹配中文颜色
    match = re.search(color_pattern, text)

    # 如果找到颜色，返回找到的颜色；否则返回 None
    if match:
        return match.group()
    return None


def extract_clothing_materials(text):
    # 常见衣服材质的关键词列表
    material_keywords = [
        "棉", "麻", "丝绸", "羊毛", "涤纶", "尼龙", "皮革", "亚麻", "纤维", "真丝", "针织",
        "锦纶", "人造革", "混纺", "雪纺", "牛仔", "绒布", "毛呢", "氨纶", "羊绒",
        "羊驼毛", "开司米", "仿皮", "合成纤维", "莫代尔", "粘胶", "天丝", "竹纤维",
        "丝光棉", "超细纤维", "太空棉", "醋酸纤维", "铜氨丝", "氯纶", "丙纶",
        "莱卡", "法兰绒", "貂绒", "羽绒", "人造丝", "毛皮", "绸缎", "植鞣革",
        "麂皮", "羊皮", "驼绒", "雪尼尔", "莫桑比克棉", "汉麻", "混纤", "苎麻",
        "烫金布", "镂空蕾丝", "丝网"
    ]
    # 提取材质相关的词汇
    extracted_materials = [material for material in material_keywords if material in text]

    # 将结果以逗号分隔的纯文本形式返回
    if len(extracted_materials) > 0:
        return '，'.join(extracted_materials)
    else:
        return None


def extract_clothing_styles(text):
    # 常见服装风格的关键词列表
    style_keywords = [
        "休闲", "商务", "优雅", "复古", "朋克", "甜美", "波西米亚", "文艺", "街头",
        "简约", "运动", "学院", "中性", "欧美", "韩版", "古典", "华丽", "民族",
        "性感", "清新", "雅痞", "摩登", "田园", "哥特", "都市", "潮流", "极简",
        "洛丽塔", "宫廷", "前卫", "嘻哈", "英伦", "民族风"
    ]

    # 提取风格相关的词汇
    extracted_styles = [style for style in style_keywords if style in text]

    # 将结果以逗号分隔的纯文本形式返回
    if len(extracted_styles) > 0:
        return '，'.join(extracted_styles)
    else:
        return None


def extract_season_with_single_match(text):
    # 定义匹配季节和组合季节的正则表达式
    season_pattern = r'(春夏|春秋|春冬|夏秋|秋冬|春|夏|秋|冬)'

    # 使用正则表达式查找第一个匹配的季节
    match = re.search(season_pattern, text)

    # 如果有匹配结果，返回匹配的字符串；否则返回 None
    if match:
        return match.group()
    return None


def contains_2019_and_new_with_none(text):
    # 检查是否包含 "2019" 和 "新款"
    if "2019" in text and "新款" in text:
        return "是的"
    elif "2019" not in text and "新款" not in text:
        return None
    return "不是"


def convert_to_jsonl(qa_df, output_path):
    # Function to convert the QA pairs to the desired JSONL format and save
    jsonl_data = []

    # Iterate through the dataframe and create the required JSONL format
    for _, row in qa_df.iterrows():
        image_path = f"../../user_data/datasets/image/{row['image']}"
        entry = {
            "query": f"<image>{row['question']}",
            "response": row['answer'],
            "images": [image_path],
        }
        jsonl_data.append(entry)

    # Save the data to a jsonl file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Define question templates
prompt_templates = {
    "descriptive": "请对给定的图片进行描述。",
    "detail1": "这件{}是什么颜色的？",
    "detail2": "这件{}是什么材质的？",
    "detail3": "这件{}是什么风格的？",
    "functionality": "这款{}适合什么季节穿？",
    "comparison": "这款{}适合哪种身材的女生穿？",
    "new_style": "这款{}是2019年的新款吗？",
}


# Initialize list to store generated QA pairs
qa_pairs = []
file_path = '../user_data/datasets/train_annotation.csv'
data = pd.read_csv(file_path, sep="\t", on_bad_lines='skip', engine="python")


# Iterate through the data and generate QA pairs
for index, row in data.iterrows():
    image = row['image']
    text = row['text']

    clothing_type = find_clothing_type(text)
    if clothing_type is None:
        clothing_type = text
        if np.random.random() > 0.1:
            clothing_type = "商品"

    # Descriptive QA pair
    qa_pairs.append({
        "question": prompt_templates["descriptive"],
        "answer": f"{text}",
        "image": image
    })

    # 颜色
    color = find_chinese_color(text)
    if color is not None:
        # Detail QA pair
        qa_pairs.append({
            "question": prompt_templates["detail1"].format(clothing_type),
            "answer": f"这件{clothing_type}的颜色是{color}。",
            "image": image
        })

    # 材质
    material = extract_clothing_materials(text)
    if material is not None:
        qa_pairs.append({
            "question": prompt_templates["detail2"].format(clothing_type),
            "answer": f"这件{clothing_type}的材质是{material}。",
            "image": image
        })

    # 风格
    style = extract_clothing_styles(text)
    if style is not None:
        qa_pairs.append({
            "question": prompt_templates["detail3"].format(clothing_type),
            "answer": f"这件{clothing_type}的风格是{style}。",
            "image": image
        })

    # 季节
    season = extract_season_with_single_match(text)
    if season is not None:
        # Functionality QA pair
        if len(season) == 1:
            season = f"{season}季"
        else:
            season = f"{season}季节"
        qa_pairs.append({
            "question": prompt_templates["functionality"].format(clothing_type),
            "answer": f"这款{clothing_type}适合在{season}穿。",
            "image": image
        })

    # 是否新款
    new_2019 = contains_2019_and_new_with_none(text)
    if new_2019 is not None and np.random.random() > 0.2:
        if new_2019 == "是的":
            my_answer = f"这款{clothing_type}是2019年的新款"
        elif new_2019 == "不是":
            my_answer = f"这款{clothing_type}不是2019年的新款"
        qa_pairs.append({
            "question": prompt_templates["new_style"].format(clothing_type),
            "answer": my_answer,
            "image": image
        })


# Convert the QA pairs into a DataFrame
qa_df = pd.DataFrame(qa_pairs)

# 使用 sklearn 的 train_test_split 将数据划分为训练集和验证集
train_df, val_df = train_test_split(qa_df, test_size=0.2, random_state=42)

os.makedirs("../user_data/datasets/qa_pairs", exist_ok=True)
convert_to_jsonl(train_df, "../user_data/datasets/qa_pairs/train.jsonl")
convert_to_jsonl(val_df, "../user_data/datasets/qa_pairs/val.jsonl")

print("Generated QA pairs saved to JSONL files.")
