from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
import torch
import os

def convert_model():
    print("Initializing config...")
    # Giảm kích thước config để phù hợp với bộ nhớ
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=2048,  # Giảm xuống
        intermediate_size=5504,  # Giảm xuống
        num_hidden_layers=16,  # Giảm xuống
        num_attention_heads=16,  # Giảm xuống
        num_key_value_heads=16,  # Giảm xuống
        hidden_act="silu",
        max_position_embeddings=2048,  # Giảm xuống
        rms_norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    
    print("Creating model from config...")
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.float16,
        load_in_8bit=True
    )
    
    print("Creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/llama-tokenizer",
        trust_remote_code=True
    )
    
    print("Saving model and tokenizer...")
    output_dir = "models/llama-3.2-1b-instruct-hf"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sử dụng các options để giảm bộ nhớ khi lưu
    model.save_pretrained(
        output_dir,
        max_shard_size="200MB",  # Chia thành các file nhỏ hơn
        safe_serialization=True,
        low_cpu_mem_usage=True
    )
    tokenizer.save_pretrained(output_dir)
    
    print(f"Successfully converted and saved model to {output_dir}")

if __name__ == "__main__":
    convert_model()