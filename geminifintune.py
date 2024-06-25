import csv  # Import the csv module to work with CSV files
# Define a function that converts a CSV file to a list of dictionaries
def csv_to_list_of_dicts(csv_file):
    data = []  
    with open(csv_file, 'r') as file:
        
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)
        
        # Iterate over each row in the CSV file
        for row in csv_reader:  
            data.append({'text_input': row['comment'], 
                          'output': row['label']})
    # Return the list of dictionaries
    return data  
# Replace 'your_dataset.csv' with the path to your CSV file
fine_tuning_data = csv_to_list_of_dicts('合并版comments_100_taged.csv')

print(fine_tuning_data[:2])

# Importing the generativeai module from the google package.
import google.generativeai as genai
# Configuring the generativeai module to use gRPC (Google Remote Procedure Call) as the transport protocol.
genai.configure(transport='grpc')

# Assigning a name to the fine-tuned model.
my_finetuned_model_name = "model300sample"
# Creating a tuned model operation with specified parameters.
operation = genai.create_tuned_model(
    # Setting the source model for tuning.
    source_model= 'models/gemini-1.0-pro-001',
    # Providing the training data for fine-tuning.
    training_data = fine_tuning_data,
    # Assigning an ID to the fine-tuned model.
    id = my_finetuned_model_name,
    # Setting the number of epochs for training.
    epoch_count = 100,
    # Specifying the batch size for training.
    batch_size=4,
    # Setting the learning rate for training.
    learning_rate=0.001,
)

model = genai.get_tuned_model(f'tunedModels/{my_finetuned_model_name}')

model.state
operation.metadata

import pandas as pd
# Importing the seaborn library with an alias sns for ease of use.
import seaborn as sns
# Retrieving the result of the operation, which is the fine-tuned model.
model = operation.result()
# Creating a DataFrame from the snapshots of the tuning task in the model.
snapshots = pd.DataFrame(model.tuning_task.snapshots)
# Creating a line plot using seaborn, with epoch on the x-axis and mean loss on the y-axis.
sns.lineplot(data=snapshots, x='epoch', y='mean_loss')

from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
import re
import time
import random

# 加载微调后的模型
model = genai.GenerativeModel(model_name=f'tunedModels/model300sample')

# 假设你的测试数据集是一个CSV文件，包含'input'（输入文本）和'true_output'（实际标签）两列
test_data = pd.read_csv('weibo_300_taged.csv')

# 用于存储预测结果的列表


def extract_numbers(text):
    # 使用正则表达式提取数字
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])  # 返回第一个匹配的数字，并转换为整数
    else:
        return None  # 如果没有找到数字，返回None或适当的占位符

predictions = []

for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
    input_text = row['content']
    prompt = f"{input_text} 只输出数字，千万不要输出中文、英文、特殊符号等。"
    try:
        response = model.generate_content(prompt)
        if len(response.parts) == 1:
            generated_output = extract_numbers(response.parts[0].text)
        else:
            # 连接所有部分并从中提取数字
            combined_text = " ".join(part.text for part in response.parts if 'text' in part)
            generated_output = extract_numbers(combined_text)
        if generated_output is not None:
            predictions.append(generated_output)
        else:
            print("No number extracted, using placeholder.")
            predictions.append(-1)  # 使用-1或其他适当的占位符表示无法提取数字
    except Exception as e:
        print(f"Error generating content: {e}")
        time.sleep(60)  # 等待一分钟后重试
        continue
    # 添加随机延时
    time.sleep(random.randint(6, 10))

test_data['predicted_output'] = predictions

# 计算性能指标
precision = precision_score(test_data['label'], test_data['predicted_output'], average='macro', zero_division=0)
recall = recall_score(test_data['label'], test_data['predicted_output'], average='macro', zero_division=0)
f1 = f1_score(test_data['label'], test_data['predicted_output'], average='macro', zero_division=0)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
