import logging
import os
import yaml
from typing import Dict, Any

logger = logging.getLogger(__name__)

def train_reviewer(config_path: str):
    """
    Train a specialized Reviewer model using LoRA fine-tuning.
    Attempts to import Unsloth for ultra-fast training, otherwise falls back
    to standard HuggingFace Transformers SFTTrainer.
    """
    logger.info(f"Loading SFT training configurations from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
    dataset_dir = config.get("dataset_dir", "data/training/reviewer")

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        import torch

        logger.info("Initializing Unsloth FastLanguageModel SFT...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None, # Auto-detect
            load_in_4bit=True,
        )

        # Apply LoRA PEFT
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        logger.info("PEFT adapter successfully attached using Unsloth.")

    except ImportError:
        logger.warning("Unsloth not found. Falling back to standard HuggingFace Transformers + PEFT...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_4bit=True if torch.cuda.is_available() else False
        )
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    # Load SFT Dataset
    from datasets import load_dataset
    dataset = load_dataset(
        "json", 
        data_files={
            "train": os.path.join(dataset_dir, "train.jsonl"),
            "validation": os.path.join(dataset_dir, "val.jsonl")
        }
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="messages",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            logging_steps=1,
            optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs/reviewer-lora",
        ),
    )

    logger.info("Starting trainer execution loop...")
    trainer.train()
    
    # Save adapter
    model.save_pretrained("outputs/reviewer-lora-final")
    tokenizer.save_pretrained("outputs/reviewer-lora-final")
    logger.info("Reviewer PEFT adapter saved to outputs/reviewer-lora-final")


def train_reranker(config_path: str):
    """
    Train a custom Reranker using cross-encoders based on BAAI/bge-reranker-v2-m3.
    """
    logger.info("Initializing CrossEncoder Reranker training sequence...")
    from sentence_transformers import InputExample, CrossEncoder
    from torch.utils.data import DataLoader

    # Load default data
    dataset_file = "data/training/reranker/triplets.jsonl"
    if not os.path.exists(dataset_file):
        logger.error(f"Reranker dataset file not found at {dataset_file}")
        return

    train_examples = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # CrossEncoder expects pairs with positive/negative labels
            train_examples.append(InputExample(texts=[data["query"], data["positive"]], label=1.0))
            train_examples.append(InputExample(texts=[data["query"], data["negative"]], label=0.0))

    model = CrossEncoder("BAAI/bge-reranker-v2-m3", num_labels=1)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

    logger.info("Starting reranker fine-tuning loop...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=3,
        warmup_steps=10,
        output_path="outputs/reranker-ft-final"
    )
    logger.info("Reranker fine-tuning complete! Model saved to outputs/reranker-ft-final")
