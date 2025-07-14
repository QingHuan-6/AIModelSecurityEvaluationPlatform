# PETAL

This official repository contains the source code and scripts for the paper **'Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models'**, accepted by [USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25). 

In this paper, we propose **PETAL**: a label-only membership inference attack based on **PE**r-**T**oken sem**A**ntic simi**L**arity. Specifically, PETAL leverages token-level semantic similarity to approximate output probabilities and subsequently calculate the perplexity. It finally exposes membership based on the common assumption that members are 'better' memorized and have smaller perplexity. Empirically, our PETAL performs better than the extensions of existing label-only attacks against personalized LLMs and even on par with other advanced logit-based attacks across all metrics.

## Main Content
- [Setup](#Setup)
- [Data](#Data)
- [Evaluation](#Evaluation)


## Getting Started

### Setup

Our code has been tested on Linux (a server with NVIDIA A6000 GPUs, each with 48GB memory) with Python 3.9.20, CUDA 12.1, PyTorch 2.0.1

To set up the environment, follow these three steps:

1. Download and unzip the package 
```bash
tar -xvf PETAL-USENIX-Artifacts.tar
cd PETAL-USENIX-Artifacts
```

2. Install CUDA 12.1, pytorch 2.0.1, python 3.9 within a `conda` virtual environment.
```bash
conda create -n PETAL python=3.9
conda activate PETAL
pip install numpy==1.23.0 torch==2.0.1
```
3. Run the following command to install the other required packages listed in the `requirements.txt` file in the current directory:
```bash
pip install -r requirements.txt
```

4. Run the following Python script to check if the GPU and CUDA environment are correctly recognized and available for use:

   ```python
   import torch
   
   print(torch.__version__)
   print(torch.version.cuda)
   print(torch.cuda.is_available())
   ```

   If `torch.cuda.is_available()` returns `True`, the environment is ready. 

### Data
- The source code will automatically download the required datasets from Hugging Face, so there is no need to download them separately. 
- You can also modify the code in utils.py to make it adaptable for future datasets (or other datasets not used in this paper) 😊.

### Evaluation
1. Use the following command to run PETAL:
```bash
python run.py --gpu_ids 0 --target_model pythia-6.9b --surrogate_model gpt2-xl --data WikiMIA --length 32
```
2. We also provide many commands in the **scripts** folder to help reproduce the results presented in our paper 😊.


## Acknowledgements

Our code is built upon the official repositories of [Detecting Pretraining Data from Large Language Models](https://github.com/swj0419/detect-pretrain-code) (Shi et al., ICLR 24). We sincerely appreciate their valuable contributions to the community.
