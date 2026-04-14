import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import torch

# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_PATH = "data/finetune_dataset.json"
OUTPUT_DIR = "models/lora-adapter"
MAX_SEQ_LENGTH = 2048

# ---------------------------
# Load Tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# ---------------------------
# Quantization Configuration (QLoRA)
# ---------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ---------------------------
# Load Base Model
# ---------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# ---------------------------
# LoRA Configuration
# ---------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------
# Load Dataset
# ---------------------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_prompt(example):
    prompt = (
        f"<s>[INST] {example['instruction']} [/INST] "
        f"{example['output']}</s>"
    )
    return {"text": prompt}

dataset = dataset.map(format_prompt)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# ---------------------------
# Training Arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="none",
)

# ---------------------------
# Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    ),
)

# ---------------------------
# Train Model
# ---------------------------
trainer.train()

# ---------------------------
# Save LoRA Adapter
# ---------------------------
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ LoRA adapter saved to {OUTPUT_DIR}")