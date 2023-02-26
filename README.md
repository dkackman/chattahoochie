# chattahoochie

A code playground/scratch pad/whatever for learning about running language models. Uses huggingface and if you're familiar with stable diffusion code 
should be relatively familiar. LLMs are at a point where they will soon be as accesible as SD was in late 2022 so exciting times are coming!

## Some initial things I've leaned

- VRAM size is much more important here than SD
- Models can be run at 8 bit precsion with little loss of fidelity. The [bitsandbytes library](https://github.com/TimDettmers/bitsandbytes) enables this.
- This allows large models to fit into less memory. 
- https://huggingface.co/PygmalionAI/pygmalion-6b can run in 16GB performs pretty well. 
- Models with less then 6B paramters are pretty nonsensical. They are good for quick testing though.
- [Deepspeed](https://github.com/microsoft/DeepSpeed) and [flexgen](https://github.com/FMInference/FlexGen) look very interesting for LLM comrpession and optimization on smaller GPUs.
- I'm working with a NVIDIA 3090 with 24GB VRAM and the LLMs I've played with are starting to be coherent, though can still drift into the nonsensical.

## Fun LLMs I've found so far

- [The 7B paramters versions of bloom](https://huggingface.co/bigscience) can run on commodity hardware and are fun. 
- [Pygmalion 6B](https://huggingface.co/PygmalionAI/pygmalion-6b) also seems to perform well.

## Hello Chat

Chat apps work by building up an array of strings which represent the ocnversaiton. The entire conversation is sent to the model each time, 
which is how it knows the context. The longer the larger the context the more VRAM is needed.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
```
