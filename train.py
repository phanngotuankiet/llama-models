from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import logging
import os
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DollyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = f"{item['instruction']}{item['response']}"
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": encodings["input_ids"][0].clone()
        }

def train():
    try:
        logger.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            "models/llama-3.2-1b-instruct-hf",
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

        model.train()
        for param in model.parameters():
            param.requires_grad = True

        tokenizer = AutoTokenizer.from_pretrained(
            "models/llama-3.2-1b-instruct-hf",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Preparing model for training...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

        logger.info("Loading dataset...")
        with open("models/dolly_data/training.json", "r") as f:
            train_data = json.load(f)

        dataset = DollyDataset(train_data, tokenizer)
        
        train_size = int(0.9 * len(dataset))
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, len(dataset) - train_size]
        )

        logger.info(f"Dataset size: {len(dataset)} (Train: {len(train_dataset)}, Val: {len(val_dataset)})")

        training_args = TrainingArguments(
            output_dir="models/checkpoints",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=32,
            learning_rate=2e-4,
            fp16=False,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            warmup_steps=100,
            load_best_model_at_end=True,
            report_to="none",
            logging_dir="./logs",
            gradient_checkpointing=False,
            max_grad_norm=0.3,
            optim="adamw_torch",
            max_steps=2000,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        logger.info("Starting training...")
        trainer.train()

        logger.info("Saving final model...")
        trainer.save_model("models/lora")
        
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train()