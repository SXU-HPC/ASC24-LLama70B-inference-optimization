"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase,PreTrainedTokenizerFast)
from tqdm import tqdm
from vllm import LLM, SamplingParams

def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer,
    trust_remote_code: bool,
) -> float:
    # print(model)
    llm = LLM(model, trust_remote_code=trust_remote_code, tensor_parallel_size=6,pipeline_parallel_size=2)
    tokenizer.pad_token = tokenizer.eos_token
    llm.set_tokenizer(tokenizer)

    input_num_tokens = []
    output_num_tokens = []
    start = time.perf_counter()
    for i in tqdm(range(len(requests))):
        prompt, prompt_len, output_len = requests[i]
        # Generate the sequences.
        input_ids = tokenizer(prompt, return_tensors="pt",
                              padding=True).input_ids
        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,  # 控制生成文本的随机性，1.0 表示不随机
            top_p=1.0,  # 核采样的累积概率，1.0 表示总是选择最可能的词
            max_tokens=output_len  # 生成文本的最大长度
        )
        llm_outputs = llm.generate(
            prompts=prompt,
            sampling_params=sampling_params,
            use_tqdm=True
        )

        # Include the decoding time.
        tokenizer.decode(llm_outputs[0], skip_special_tokens=True)
        input_num_tokens.append(len(input_ids[0]))
        output_num_tokens.append(len(llm_outputs[0]))


    end = time.perf_counter()
    return end - start, input_num_tokens, output_num_tokens



def main(args: argparse.Namespace):
    # print(args)
    random.seed(args.seed)
    # Sample the requests.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_samples)]

    else:
        with open(args.dataset) as f:
            requests = json.load(f)

    if args.num_samples is not None:
        requests = requests[0:args.num_samples]

    elapsed_time, input_num_tokens, output_num_tokens = run_vllm(requests, args.model, tokenizer,  args.trust_remote_code)
    prompt_num_tokens = sum(prompt_len for prompt_len in input_num_tokens)
    total_num_tokens = sum(output_len for output_len in output_num_tokens)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
          f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
          f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
          f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="./model/models--meta-llama--Llama-2-70b-hf/snapshots/model")
    parser.add_argument("--tokenizer", type=str, default="./model/models--meta-llama--Llama-2-70b-hf/snapshots/model")
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for each request")
    parser.add_argument("--output-len", type=int, default=None, help="Output length for each request")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of first few samples used for inference test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    args = parser.parse_args()
    
    # print(args.input_len)
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None
        assert args.output_len is None

    main(args)
