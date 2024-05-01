# Evaluation
## Code for evaluating models using lm-evaluation-harness

This folder contains a copy for lm-evaluation-harness for reference. After installing from source following the [README] of the original repository, we show below an example command to evaluate our models. The same commands work for the base models and instruction-tuned models for both Mamba and Pythia.

For evaluating Mamba base models and instruction-tuned checkpoints:

```
$ lm_eval --model mamba_ssm --tasks hellaswag --model_args pretrained=/path/to/pythia_checkpoint --device cuda:0 --batch_size 32 --num_fewshot <if_n_shot_eval> --output_path /path/to/output.jsonl
```

Similarly to evaluate Pythia base models and instruction-tuned checkpoints:
```
$ lm_eval --model hf --tasks hellaswag --model_args pretrained=/path/to/pythia_checkpoint --device cuda:0 --batch_size 32 --num_fewshot <if_n_shot_eval> --output_path /path/to/output.jsonl
```
The arguments above can be changed are as follows:

1. --model: Type of model architecture (`mamba_ssm` for Mamba and `hf` for pythia)
2. --tasks: Single task or comma (`,`) separated list of tasks to evaluate
3. --model_args pretrained: Checkpoint of the model (can also provide HF model names here or path to folder with the saved model file)
4. --batch_size: Batch size to use during evaluation. This can be changed depending on the compute available
5. --num_fewshot: Give a number if you want to do few-shot evaluation
6. --output_path: Path of the output file where the results will be saved