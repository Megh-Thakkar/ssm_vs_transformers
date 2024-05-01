import os
import argparse
import torch
from datasets import load_dataset
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer

def main(args):

    num_train_epochs = 3
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2
    gradient_accumulation_steps = 6

    # Save checkpoint every X updates steps
    if 'dolly' in args.dataset.lower():
        save_steps = 600
    elif 'lima' in args.dataset.lower():
        save_steps = 80

    # Log every X updates steps
    if 'dolly' in args.dataset.lower():
        logging_steps = 100
    elif 'lima' in args.dataset.lower():
        logging_steps = 10

    learning_rate = args.lr
    max_seq_length = 2048


    # Default settings here

    fp16 = False
    bf16 = True
    gradient_checkpointing = True
    max_grad_norm = 0.3
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "constant"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    packing = False
    device_map = {"": 0}
    dataset = load_dataset(args.dataset, split="train")
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)


    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    def transform_conversation_dolly(example):
        content = [{'content': example['instruction'], 'role': 'user'}, {'content': example['response'], 'role': 'assistant'}]
        return {"text": tokenizer.decode(tokenizer.apply_chat_template(content, chat_template=tokenizer.default_chat_template))}

    def transform_conversation_lima(example):
        content = [{'content': example['conversations'][0], 'role': 'user'}, {'content': example['conversations'][1], 'role': 'assistant'}]
        return {"text": tokenizer.decode(tokenizer.apply_chat_template(content, chat_template=tokenizer.default_chat_template))}

    if 'dolly' in args.dataset.lower():
        dataset = dataset.map(transform_conversation_dolly)

    elif 'lima' in args.dataset.lower():
        small_ds = []

        for d in dataset:
            if len(d['conversations']) == 2:
                small_ds.append(d)

        small_ds_dataset = datasets.Dataset.from_list(small_ds)
        dataset = small_ds_dataset.map(transform_conversation_lima)

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    trainer.train()
    trainer.model.save_pretrained(os.path.join(args.output_dir, args.new_model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-2.8b-deduped")
    parser.add_argument("--new_model", type=str, default="pythia-2.8b-deduped-dollybricks_lr_1e-5")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dataset", type=str, default="abcd")
    parser.add_argument("--output_dir", type=str, default="/network/scratch/m/megh.thakkar/outputs/mamba/pythia-2.8b-dollybricks_lr_5e-5")
    args = parser.parse_args()

    if 'dolly' in args.dataset:
        args.dataset = "databricks/databricks-dolly-15k"
    elif 'lima' in args.dataset.lower():
        args.dataset = "GAIR/lima"

    main(args)