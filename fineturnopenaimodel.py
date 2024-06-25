#use code to finetune
from openai import OpenAI
import time

# 设置API密钥
openai_api_key = ""
client = OpenAI(api_key=openai_api_key)

file_path = '/content/train_data_chat_2500_comments_hebing.jsonl'

def ensure_string_content(line):
    try:
        obj = json.loads(line)
        if "messages" in obj:
            for message in obj["messages"]:
                if "content" in message and not isinstance(message["content"], str):
                    message["content"] = str(message["content"])
        return json.dumps(obj, ensure_ascii=False) + "\n"
    except json.JSONDecodeError:
        return line

# 读取并修复文件内容
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

with open(file_path, 'w', encoding='utf-8') as file:
    for line in lines:
        corrected_line = ensure_string_content(line)
        file.write(corrected_line)
with open(file_path, 'rb') as file:
    file_response = client.files.create(
        file=file,
        purpose='fine-tune'
    )
train_file_id = file_response.id  # 直接访问属性
print(f"文件上传成功，文件ID：{train_file_id}")

# 创建微调作业
fine_tune_response = client.fine_tuning.jobs.create(
    training_file=train_file_id,
    model="gpt-3.5-turbo"
)
fine_tune_id = fine_tune_response.id  # 直接访问属性
print(f"微调作业已创建，作业ID：{fine_tune_id}")

# 等待微调完成
status = 'pending'
while status != 'succeeded':
    fine_tune_response = client.fine_tuning.jobs.retrieve(fine_tune_id)
    status = fine_tune_response.status  # 直接访问属性
    if status == 'succeeded':
        fine_tuned_model = fine_tune_response.fine_tuned_model  # 直接访问属性
        print(f"微调成功，微调模型ID：{fine_tuned_model}")
    elif status == 'failed':
        raise Exception("微调失败")
    else:
        print(f"微调状态：{status}，等待60秒...")
        time.sleep(60)
