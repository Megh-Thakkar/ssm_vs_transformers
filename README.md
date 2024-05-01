# Can SSMs replace Transformers?
## A deep-dive into the performance of Mamba in standard NLP settings

[Shravan Nayak](mailto:perampalli.shravan.nayak@umontreal.ca), [Megh Vipul Thakkar](mailto:megh.vipul.thakkar@umontreal.ca), [Kairavi Bajaj](mailto:kairavi.bajaj@umontreal.ca)

**Université de Montréal**


Code for the course project benchmarking the capabilities of Mamba as compared to transformer-based models (particularly Pythia) across various evaluation settings such as instruction-tuning, few-shot learning, and multilinguality.

We have divided the code repository into three folders for easier understanding and usage. Each of the folders contains the corresponding README files as well.

1. [mamba_training]: Code for instruction-tuning the Mamba models. We also provided the reformatted dataset json files compatible with the code.
2. [pythia_training]: Training script for instruction-tuning Pythia models on the same datasets as Mamba.
3. [evaluation]: Contains a copy of the lm-evaluation-harness and a README detailing how to use it for the Mamba and Pythia checkpoints generated from the above training runs.

We also provide the [requirements] file to setup the environment used. The environment can be setup by using 

```
$ pip install -r requirements.txt
```


## Code Acknowledgements

We would like to thank the following for making their code open-source, easy to use, and accessible. This work would not have been possible without you :)

1. mamba_chat: [https://github.com/redotvideo/mamba-chat](https://github.com/redotvideo/mamba-chat)
2. Huggingface transformers: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
3. lm-evaluation-harness: [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)