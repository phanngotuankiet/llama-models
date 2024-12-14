from llama_cpp import Llama

# Tắt GPU layers để chạy trên CPU
llm = Llama(
    model_path="models/llama-3.2-1b-instruct.gguf",
    lora_path="models/lora.gguf",
    # tao nghĩ context length là ám chỉ độ dài câu trả lời
    n_ctx=1024,
    # n_threads=8: số lượng thread chạy trên CPU
    n_threads=8,
    # n_gpu_layers=32: số lượng layer chạy trên GPU
    n_gpu_layers=32,
    verbose=True
)

# Chat loop
while True:
    user_input = input("User: ")
    if user_input == "quit":
        break
        
    print("Đang xử lý...")
    response = llm.create_completion(
        prompt=user_input,
        max_tokens=2048,
        temperature=0.7,
        stop=["User:", "\n"]
    )
    print("Đã xử lý xong")
    print("Assistant:", response['choices'][0]['text'])