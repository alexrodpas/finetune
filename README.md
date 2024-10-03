# 👩‍💻 Fine-tune Phi-3-mini model to generate Python Code

## phi3-mini-python

**Phi3-mini** model fine-tuned on the **python_code_instructions_18k_alpaca Code instructions dataset** using **LoRA** or **QLoRA** with [PEFT](https://github.com/huggingface/peft) and bitsandbytes library.

## With LoRA

Notebook: `phi3-finetune-lora-pycoder.ipynb`

[Model in Huggingface](https://huggingface.co/alexrodpas/phi3-mini-4k-lora-python-code-18k)

[Adapter in Huggingface](https://huggingface.co/alexrodpas/phi3-mini-LoRA)

## With QLoRA

Notebook: `phi3-finetune-qlora-pycoder.ipynb`

[Model in Huggingface](https://huggingface.co/alexrodpas/phi3-mini-4k-qlora-python-code-18k)

[Adapter in Huggingface](https://huggingface.co/alexrodpas/phi3-mini-QLoRA)

## Problem description

Our goal is to fine-tune the pretrained model, Phi3-mini. amodel with 3.8B parameters, using both the PEFT method, and **LoRA** or a 4-bit quantization **QLoRA** to produce a Python coder.Then we evaluate the performance of both models. We fine-tune using a NVIDIA A100 GPU to get better performance. Alternatively, you can try out to run it on for example a T4 in Google Colab by adjusting some parameters (like batch size) to reduce memory consumption.

## Dataset

For our fine-tuning process, we use this [dataset](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) that contains about 18,000 examples where the model is asked to build a Python code that solves a given task. This is an subset of this other [original dataset](https://huggingface.co/datasets/sahil2801/code_instructions_120k) from which only the Python language examples are selected. Each row contains the description of the task to be solved, an example of data input to the task if applicable, and the generated code fragment that solves the task is provided.


## Base model

[Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

The Phi-3-Mini-4K-Instruct is a 3.8B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties. The model belongs to the Phi-3 family with the Mini version in two variants of the context length (in tokens) that it can support: 4K and 128K.

The model has underwent a post-training process that incorporates both supervised fine-tuning and direct preference optimization for the instruction following and safety measures. When assessed against benchmarks testing common sense, language understanding, math, code, long context and logical reasoning, Phi-3 Mini-4K-Instruct showcased a robust and state-of-the-art performance among models with less than 13 billion parameters.


### Example of usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "alexrodpas/phi3-mini-4k-qlora-python-code-20k"
device_map = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto", device_map=device_map)

input="'Create a function to calculate the sum of a sequence of integers.\n Input: [1, 2, 3, 4, 5]'"

# Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# Prepare the prompt or input to the model
prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": input}], tokenize=False, add_generation_prompt=True)
# Run the pipe to get the answer
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95,
                   max_time= 180)
print(outputs[0]['generated_text'][len(prompt):].strip())

```
