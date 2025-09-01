
import os
from src import prompts
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from loguru import logger

def formatting_prompts_func(examples, eos_token):
    questions = examples["question"]
    schemas = examples["schema"]
    cots = examples["cot"]
    intents = examples["intent"]
    texts = []

    for question, schema, cot, intent in zip(questions, schemas, cots, intents):
        text = prompts.INTENT_INSTRUCTION.format(question, schema) + \
            prompts.INTENT_TRAIN.format(cot, intent) + eos_token
        texts.append(text)

    return {"text": texts}

def main(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    dataset = dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer.eos_token), 
        batched=True, 
        remove_columns=dataset.column_names
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=32,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_steps=50,
            num_train_epochs=5,
            learning_rate=5e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="./outputs",
            save_strategy="steps",
            save_steps=1,
            eval_strategy="no",
            save_total_limit=2,
            dataloader_num_workers=32
        )
    )

    logger.info("Training started")
    trainer.train()
    
    os.makedirs(args.save_directory, exist_ok=True)
    trainer.model.save_pretrained(args.save_directory)
    tokenizer.save_pretrained(args.save_directory)
    logger.info(f"Model saved to {args.save_directory}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.1-8B-unsloth-bnb-4bit")
    parser.add_argument("--max_seq_length", type=int, default=5500)
    parser.add_argument("--dataset_path", type=str, default="./data/train_dataset.jsonl")
    parser.add_argument("--save_directory", type=str, default="./outputs")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)