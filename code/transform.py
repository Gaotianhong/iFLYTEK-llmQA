import os
import re
import json
import base64
import pandas as pd

from PIL import Image
from io import BytesIO
from tqdm import tqdm


# Images train:10214 valid:2554 test:1884
datapath = '../user_data/datasets'


def find_unused_images():
    # Load all image names in the 'images' folder
    image_folder = os.path.join(datapath, 'image')
    all_image_names = set(os.listdir(image_folder))

    # Function to extract image names from JSONL files
    def extract_image_names(jsonl_path):
        image_names = set()
        with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                data = json.loads(line)
                image_ids = data.get('image_ids', [])
                image_names.add(image_ids[0])
        return image_names

    # Paths to the train and valid JSONL files
    train_jsonl_path = os.path.join(datapath, 'train_texts.jsonl')
    valid_jsonl_path = os.path.join(datapath, 'valid_texts.jsonl')

    # Extract image names used in train and valid sets
    train_image_names = extract_image_names(train_jsonl_path)
    valid_image_names = extract_image_names(valid_jsonl_path)

    # Combine the image names from both train and valid sets
    used_image_names = train_image_names.union(valid_image_names)

    # Find images in the 'images' folder that are not in train or valid
    unused_images = all_image_names - used_image_names

    return list(unused_images)


def generate_jsonl_and_tsv():
    total_imgs = len(os.listdir(os.path.join(datapath, 'image')))
    print(f'total images: {total_imgs}')

    # Load the CSV file
    file_path = os.path.join(datapath, 'train_annotation.csv')
    df = pd.read_csv(file_path, sep="\t", on_bad_lines='skip', engine="python")

    # Add text_id starting from 1
    df['text_id'] = range(1, len(df) + 1)

    # Rename the 'image' column to 'image_ids' and format as list
    df['image_ids'] = df['image'].apply(lambda x: [x])

    # Select only the necessary columns
    df_formatted = df[['text_id', 'text', 'image_ids']]

    # Split data into 80% training and 20% validation
    total_rows = len(df_formatted)
    train_size = int(total_rows * 0.8)

    # Shuffle the data
    df_shuffled = df_formatted.sample(frac=1, random_state=42)

    # Create train and validation sets
    df_train = df_shuffled[:train_size]
    df_valid = df_shuffled[train_size:]

    # Function to save dataframe to JSONL format

    def save_to_jsonl(df, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(row.to_json(force_ascii=False) + '\n')

    # Save the train and validation sets
    save_to_jsonl(df_train, os.path.join(datapath, 'train_texts.jsonl'))
    save_to_jsonl(df_valid, os.path.join(datapath, 'valid_texts.jsonl'))

    # Function to encode images to base64 and save as TSV
    def save_to_tsv(jsonl_path, tsv_path):
        line_count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file, open(tsv_path, 'w', encoding='utf-8') as tsv_file:
            lines = jsonl_file.readlines()  # Read all lines to get the total count for progress
            for line in tqdm(lines, desc=f"Processing {os.path.basename(jsonl_path)}", unit="line"):
                data = json.loads(line)
                img_name = data['image_ids'][0]
                try:
                    # Open and convert the image to base64
                    img_path = os.path.join(datapath, 'image', img_name)
                    with Image.open(img_path) as img:
                        img_buffer = BytesIO()
                        img.save(img_buffer, format=img.format)
                        byte_data = img_buffer.getvalue()
                        base64_str = base64.b64encode(byte_data).decode("utf-8")
                        # Write text_id and base64_str to TSV
                        tsv_file.write(f"{img_name}\t{base64_str}\n")
                        line_count += 1
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")
        print(f"Total number of entries written to {os.path.basename(tsv_path)}: {line_count}")

    # Save the train and validation sets
    save_to_tsv(os.path.join(datapath, 'train_texts.jsonl'), os.path.join(datapath, 'train_imgs.tsv'))
    save_to_tsv(os.path.join(datapath, 'valid_texts.jsonl'), os.path.join(datapath, 'valid_imgs.tsv'))

    # Load the query.json file
    query_file_path = os.path.join(datapath, 'query.json')

    # Read the JSON data from the query file
    with open(query_file_path, 'r', encoding='utf-8') as file:
        query_data = json.load(file)

    # Extract the questions where 'related_image' is empty and prepare them for the test set
    test_data = []

    for query in query_data:
        if query["related_image"] == "":
            match = re.search(r'请匹配到与(.+?)最相关的图片', query["question"])
            extracted_text = match.group(1).strip()
            test_data.append({
                "text_id": total_rows + len(test_data) + 1,
                "text": extracted_text,
                "image_ids": []
            })

    # Save the test data into a JSONL file
    test_file_path = os.path.join(datapath, 'test_texts.jsonl')
    with open(test_file_path, 'w', encoding='utf-8') as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    test_img_ids = find_unused_images()
    tsv_path = os.path.join(datapath, 'test_imgs.tsv')
    with open(tsv_path, 'w', encoding='utf-8') as tsv_file:
        for img_name in tqdm(test_img_ids, desc=f"Processing {os.path.basename(tsv_path)}", unit="line"):
            try:
                # Open and convert the image to base64
                img_path = os.path.join(datapath, 'image', img_name)
                with Image.open(img_path) as img:
                    img_buffer = BytesIO()
                    img.save(img_buffer, format=img.format)
                    byte_data = img_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    # Write text_id and base64_str to TSV
                    tsv_file.write(f"{img_name}\t{base64_str}\n")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")


generate_jsonl_and_tsv()
