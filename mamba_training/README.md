# Mamba Trainer
## Code for instruction-tuning Mamba models

This folder contains the required files for instruction-tuning Mamba models. We also provide the [Dolly15k] and [LIMA] datasets in the required jsonl format compatible with the code. To run the instruction-tuning, use the following example command:
```
$ python train_mamba.py  --model state-spaces/mamba-2.8b-slimpj --tokenizer  EleutherAI/gpt-neox-20b  --learning_rate 5e-5 --batch_size  6  --gradient_accumulation_steps  2  --data_path  ./data/lima.jsonl  --num_epochs  3  --output_dir=/path/out
```
The arguments above can be changed are as follows:

1. --model: The base model to instruction-tune (you can choose any variant of Mamba here)
2. --learning_rate: Learning rate for training
3. --batch_size: Batch size used, can be changed as per the compute available
4. --gradient_accumulation_steps: Gradient accumulation steps to offset scarcity of compute
5. --data_path: Path to the dataset jsonl
6. --num_epochs: Number of epochs to train
7. --output_dir: Output directory to save the checkpoints

We currently save the model every 80 steps for LIMA and 600 steps for Dolly15k. These can be changed very easily in the [train_mamba.py] file.