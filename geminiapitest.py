import os
import time
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from sklearn.metrics import precision_score, recall_score, f1_score
import re
import random
# 配置 API
api_key = ""
genai.configure(api_key=api_key)

# 初始化模型
model = genai.GenerativeModel('gemini-1.0-pro')

# 加载测试数据集
test_data = pd.read_csv('/content/weibo_100_taged.csv')

def generate_prompt_without_context(text):
    return (
        f"请将以下文本分类为以下四类之一：生育调节（1），性传播感染的预防和管理（2），性暴力预防和性功能（3），无关（4）。仅仅标注数字代码，不要中文原文\n\n"
        f"文本：{text}\n分类："
    )

def generate_prompt_with_context(text, train_texts, train_labels):
    prompt = (
        f"请根据已知的标注数据来分类以下文本。仅仅标注数字代码，不要出现中文原文\n已知数据包括以下几组（以文本和分类对出现）：\n文本分类为以下四类之一：生育调节（1），性传播感染的预防和管理（2），性暴力预防和性功能（3），无关（4）\n"
    )
    for t, l in list(zip(train_texts, train_labels))[:5]:
        prompt += f"文本：{t}\n分类：{l}\n\n"
    prompt += f"现在，请分类以下文本：\n\n文本：{text}\n分类："
    return prompt

def extract_numbers(text):
    numbers = re.findall(r'\d+', text)
    return ''.join(numbers)

train_texts = test_data['content'].tolist()
train_labels = test_data['label'].tolist()

predictions_without_context = []
predictions_with_context = []

for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
    text = row['content']
    prompt_without_context = generate_prompt_without_context(text)
    prompt_with_context = generate_prompt_with_context(text, train_texts, train_labels)

    for _ in range(3):  # 最多尝试三次
        try:
            # 无上下文
            response_without_context = model.generate_content(prompt_without_context)
            pred_without_context = int(extract_numbers(response_without_context.text)) if response_without_context.parts else -1

            # 有上下文
            response_with_context = model.generate_content(prompt_with_context)
            pred_with_context = int(extract_numbers(response_with_context.text)) if response_with_context.parts else -1

            if pred_without_context != -1 and pred_with_context != -1:
                break
        except Exception as e:
            print(f"Attempt failed: {e}")
            time.sleep(60)  # 等待一分钟后重试

    predictions_without_context.append(pred_without_context)
    predictions_with_context.append(pred_with_context)
    time.sleep(random.randint(6, 10))

test_data['predicted_output_without_context'] = predictions_without_context
test_data['predicted_output_with_context'] = predictions_with_context

# 计算性能指标
precision_without = precision_score(test_data['label'], test_data['predicted_output_without_context'], average='macro', zero_division=0)
recall_without = recall_score(test_data['label'], test_data['predicted_output_without_context'], average='macro', zero_division=0)
f1_without = f1_score(test_data['label'], test_data['predicted_output_without_context'], average='macro', zero_division=0)

precision_with = precision_score(test_data['label'], test_data['predicted_output_with_context'], average='macro', zero_division=0)
recall_with = recall_score(test_data['label'], test_data['predicted_output_with_context'], average='macro', zero_division=0)
f1_with = f1_score(test_data['label'], test_data['predicted_output_with_context'], average='macro', zero_division=0)

# 打印结果
print("Results without context:")
print(f'Precision: {precision_without}')
print(f'Recall: {recall_without}')
print(f'F1 Score: {f1_without}')

print("Results with context:")
print(f'Precision: {precision_with}')
print(f'Recall: {recall_with}')
print(f'F1 Score: {f1_with}')
