from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
from .app import _parse_args_from_argv, get_type


def main(model_name):
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-ul2", torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

    input_string = "Answer the following question by reasoning step by step. The cafeteria had 23 apples. If they used 20 for lunch, and bought 6 more, how many apple do they have?"

    inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, max_length=200)

    print(tokenizer.decode(outputs[0]))
    # <pad> They have 23 - 20 = 3 apples left. They have 3 + 6 = 9 apples. Therefore, the answer is 9.</s>


def load_model_sharded(model_name):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with init_empty_weights():
        model = AutoModelForSeq2SeqLM.from_config(config)

    device_map = infer_auto_device_map(
        model, max_memory={0: "20GiB", "cpu": "32GiB"})

    model = load_checkpoint_and_dispatch(
        model, dtype=torch.float16, checkpoint="sharded-ul2", device_map=device_map).eval()

    return model, tokenizer


if __name__ == "__main__":
    args = _parse_args_from_argv()

    main(model_name=args.model_name)
