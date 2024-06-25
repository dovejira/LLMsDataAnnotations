#transfer the csv to jsonl to help llms to analyse
import pandas as pd
import json

def clean_content(content):
    return content.replace("#超斯基信箱#", "").strip()#cleaning data

def convert_to_chat_jsonl(df, output_path):
    # Clean the content column
    df['cleaned_content'] = df['content'].apply(clean_content)

    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            messages = [
                {"role": "system", "content": "这是一个分类聊天机器人。"},
                {"role": "user", "content": row['cleaned_content']},
                {"role": "assistant", "content": str(row['label'])}
            ]
            json_line = json.dumps({"messages": messages}, ensure_ascii=False)
            f.write(json_line + '\n')

# Load your data
human_tagged_data = pd.read_csv('/content/weibo_200_taged.csv')

# Convert and save to JSONL
convert_to_chat_jsonl(human_tagged_data, '/content/train_data_chat_200_weibo.jsonl')
