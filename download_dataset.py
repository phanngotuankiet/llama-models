from datasets import load_dataset
import json
import os

# Tạo thư mục nếu chưa tồn tại
os.makedirs("models/dolly_data", exist_ok=True)

# Xóa file training.json cũ nếu tồn tại
training_file = "models/dolly_data/training.json"
if os.path.exists(training_file):
    os.remove(training_file)
    print(f"Đã xóa file cũ: {training_file}")

# Tải dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# Convert và format data
converted_data = []
for item in dataset:
    # Tách các phần của cuộc hội thoại 
    parts = item["chosen"].split("\n\nHuman: ")
    
    for part in parts:
        if "Assistant: " in part:
            # Tách instruction và response
            conv = part.split("\n\nAssistant: ")
            if len(conv) == 2:
                instruction = conv[0].replace("Human: ", "").strip()
                response = conv[1].strip()
                
                # Format theo chuẩn của Llama
                formatted_instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                formatted_response = f"{response}<|eot_id|>"
                
                converted_data.append({
                    "instruction": formatted_instruction,
                    "response": formatted_response
                })

# Lưu data đã format
with open(training_file, "w") as f:
    json.dump(converted_data, f, indent=2)

print(f"Đã lưu {len(converted_data)} mẫu data vào {training_file}")
print("\nVí dụ mẫu đầu tiên:")
print(json.dumps(converted_data[0], indent=2))