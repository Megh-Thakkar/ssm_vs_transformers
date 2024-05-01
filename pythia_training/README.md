# Pythia Trainer
## Code for instruction-tuning Pythia models

This folder contains the required files for instruction-tuning Pythia models. We directly use huggingface datasets for using the datasets for instruction tuning. To run the instruction-tuning, use the following example command:
```
$ python train.py  --model_name EleutherAI/pythia-2.8b-deduped --lr 5e-5 --dataset lima --output_dir=/path/to/output  --new_model pythia-2.8b-instruction_tuned
```
The arguments above can be changed are as follows:

1. --model_name: The base model to instruction-tune (you can choose any variant of Pythia here)
2. --lr: Learning rate for training
3. --dataset: Dataset to use (`lima` or `dolly`)
4. --output_dir: Folder to save the checkpoints
5. --new_model: Name of the saved model

We currently save the model every 80 steps for LIMA and 600 steps for Dolly15k, same as Mamba. These can be changed very easily in the [train.py] file. It is equally easy to change the other hyperparameters at the beginning of the file, such as per_device_batch_sizes, gradient_accumulation_steps, etc based on the compute availability.
