# src/disease_prediction_app/train_lora_cpu.py
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "synthetic_data/finetune_dataset.json"
OUTPUT_DIR = "models/lora-adapter"
MAX_SEQ_LENGTH = 512

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load model on CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": "cpu"},
)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_prompt(example):
    return {
        "text": f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
    }

dataset = dataset.map(format_prompt)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# Training arguments optimized for CPU
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training completed successfully!")