from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 暂注释掉，代码没问题，用于测试下面的内容
def classify_content_default(text):
    prompt = (
        f"请将以下文本分类为以下四类之一：生育调节（1），性传播感染的预防和管理（2），性暴力预防和性功能（3），无关（4）。务必仅仅标注数字代码，一定不要出现中文原文\n\n"
        f"文本：{text}\n"
        f"分类："
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=10,
        stop=None
    )

    category = response.choices[0].message.content.strip()
    # 提取数字编码
    if '1' in category:
        return 1
    elif '2' in category:
        return 2
    elif '3' in category:
        return 3
    elif '4' in category:
        return 4
    else:
        raise ValueError(f"Unexpected category: {category}")

def clean_text(text):
    # 移除 #超斯基信箱# 内容
    text = re.sub(r'#超斯基信箱#', '', text)
    # 移除多余的空白字符
    text = ' '.join(text.split())
    return text

# 加载并清洗数据集
human_tagged_data = pd.read_csv('/content/weibo_100_taged.csv')
human_tagged_data['cleaned_content'] = human_tagged_data['content'].apply(clean_text)

# 读取清洗后的文本和标签
test_texts = human_tagged_data['cleaned_content'].tolist()
true_labels = human_tagged_data['label'].tolist()

# 确保 true_labels 是整数类型
true_labels = [int(label) for label in true_labels]

# 进行预测并计算指标（默认方法）
predictions_default = [classify_content_default(text) for text in tqdm(test_texts, desc="Default Model")]
accuracy_default = accuracy_score(true_labels, predictions_default)
recall_default = recall_score(true_labels, predictions_default, average='macro')
f1_default = f1_score(true_labels, predictions_default, average='macro')

print(f"Default Model - Accuracy: {accuracy_default}, Recall: {recall_default}, F1 Score: {f1_default}")

# 定义分类函数（有监督的方法）
def classify_content_supervised(text, train_texts, train_labels):
    prompt = (
        f"请根据已知的标注数据来分类以下文本。仅仅标注数字代码，一定不要出现中文原文\n"
        f"已知数据包括以下几组（以文本和分类对出现）：\n\n"
        f"文本分类为以下四类之一：生育调节（1），性传播感染的预防和管理（2），性暴力预防和性功能（3），无关（4）\n"

    )
    for t, l in list(zip(train_texts, train_labels))[:5]:  # 只取前5个样本，避免上下文过长
        prompt += f"文本：{t}\n分类：{l}\n\n"
    prompt += f"现在，请分类以下文本：\n\n文本：{text}\n分类："

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=10,
        stop=None

    )

    category = response.choices[0].message.content.strip()
    # 提取数字编码
    if '1' in category:
        return 1
    elif '2' in category:
        return 2
    elif '3' in category:
        return 3
    elif '4' in category:
        return 4
    else:
        raise ValueError(f"Unexpected category: {category}")

# 进行预测并计算指标（有监督的方法）
train_texts = human_tagged_data['cleaned_content'].tolist()
train_labels = human_tagged_data['label'].tolist()

# 确保 train_labels 是整数类型
train_labels = [int(label) for label in train_labels]

predictions_supervised = [classify_content_supervised(text, train_texts, train_labels) for text in tqdm(test_texts, desc="Supervised Model")]
accuracy_supervised = accuracy_score(true_labels, predictions_supervised)
recall_supervised = recall_score(true_labels, predictions_supervised, average='macro')
f1_supervised = f1_score(true_labels, predictions_supervised, average='macro')

print(f"Supervised Model - Accuracy: {accuracy_supervised}, Recall: {recall_supervised}, F1 Score: {f1_supervised}")

# 定义分类函数（微调模型）
def classify_content_finetuned(text, fine_tuned_model):
    messages = [
        {"role": "system", "content": "这是一个分类聊天机器人。"},
        {"role": "user", "content": text}
    ]
    response = client.chat.completions.create(
        model=fine_tuned_model,
        messages=messages,
        temperature=0.7,
        max_tokens=10,
        stop=None
    )
    category = response.choices[0].message.content.strip()
    # 提取数字编码
    if '1' in category:
        return 1
    elif '2' in category:
        return 2
    elif '3' in category:
        return 3
    elif '4' in category:
        return 4
    else:
        raise ValueError(f"Unexpected category: {category}")

# 进行预测并计算指标（微调模型）
fine_tuned_model = 'ft:gpt-3.5-turbo-0125:dovejira::9Yl2QWfE'
predictions_finetuned = [classify_content_finetuned(text, fine_tuned_model) for text in tqdm(test_texts, desc="Finetuned Model")]
accuracy_finetuned = accuracy_score(true_labels, predictions_finetuned)
recall_finetuned = recall_score(true_labels, predictions_finetuned, average='macro')
f1_finetuned = f1_score(true_labels, predictions_finetuned, average='macro')

print(f"Finetuned Model - Accuracy: {accuracy_finetuned}, Recall: {recall_finetuned}, F1 Score: {f1_finetuned}")
