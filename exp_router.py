import argparse
from http import client
import json
from pathlib import Path
from tqdm import tqdm
from src import prompts

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field
from collections import Counter
from openai import OpenAI
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

load_dotenv(find_dotenv())

class Answer(BaseModel):
    rationale: str = Field(description="step by step reasoning")
    label: int = Field(description="whether the nlq is retrieval(0) or machine learning(1)")

def _load_dataset(args):
    data = [json.loads(line) for line in Path(args.dataset_path).read_text().splitlines()]
    return data

def openai_inference(model_args, sampling_args, inputs: list[dict]):
    inputs = [prompts.ROUTE.format(nlq=s['question']) for s in inputs]
    
    vote = True if sampling_args['n'] > 1 else False
    client = OpenAI()
    outputs = []
    for s in inputs:
        if vote:
            responses = []
            for _ in range(sampling_args['n']):
                response = _openai_response(client, model_args, sampling_args, s['question'])
                responses.append(response)
            outputs.append(responses)
        else:
            response = _openai_response(client, model_args, sampling_args, s['question'])
            outputs.append(response)

    results = _openai_parse_outputs(inputs, outputs, vote=vote)

    return results

def _openai_response(client, model_args, sampling_args, nlq: str):
    response = client.responses.parse(
        **model_args,
        temperature=sampling_args['temperature'],
        max_output_tokens=sampling_args['max_tokens'],
        top_p=sampling_args['top_p'],
        input=[{"role": "user", "content": prompts.ROUTE.format(nlq=nlq)}],
        text_format=Answer,
    )
    return response.output_parsed

def _openai_parse_outputs(inputs, outputs, vote=False):
    index2label = {
        0: "retrieval",
        1: "machine learning"
    }
    results = []
    for example, output in zip(inputs, outputs):
        if vote:
            label_votes = [out.label for out in output if out is not None]
            if label_votes:
                predicted_label = Counter(label_votes).most_common(1)[0][0]
                human_readable_label = index2label.get(predicted_label, f"Unknown label: {predicted_label}")
            else:
                predicted_label = None
                human_readable_label = None
        else:
            try:
                predicted_label = output.label
                human_readable_label = index2label.get(predicted_label, f"Unknown label: {predicted_label}")
            except:
                predicted_label = None
                human_readable_label = None

        results.append({
            "nlq": example["question"],
            "true_label": example["label"],
            "predicted_label": predicted_label,  # Numeric/original label
            "predicted_label_text": human_readable_label,  # Human-readable label
            "total_tokens": 0
        })
    return results

def vllm_inference(model_args, sampling_args, inputs: list[dict]):
    # LLM setting
    llm = LLM(**model_args)
    json_schema = Answer.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(**sampling_args,
        guided_decoding=guided_decoding_params,
    )

    formatted_prompts = [prompts.ROUTE.format(nlq=s['question']) for s in inputs]
    outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)
    results = _vllm_parse_outputs(inputs, outputs, vote=True if sampling_args['n'] > 1 else False)

    return results

def _vllm_parse_outputs(inputs, outputs, vote=False):
    index2label = {
        0: "retrieval",
        1: "machine learning"
    }
    results = []
    total_tokens = 0
    for example, output in zip(inputs, outputs):
        # do voting
        if vote:
            label_votes = []
            for out in output.outputs:
                try:
                    pred = json.loads(out.text)
                    label_votes.append(pred["label"])
                    total_tokens += len(out.token_ids)
                except Exception as e:
                    label_votes.append(None)
            label_votes = [label for label in label_votes if label is not None]

            if label_votes:
                predicted_label = Counter(label_votes).most_common(1)[0][0]
                human_readable_label = index2label.get(predicted_label, f"Unknown label: {predicted_label}")
            else:
                predicted_label = None
                human_readable_label = None
            total_tokens += len(output.prompt_token_ids)
        else:
            try:
                pred = json.loads(output.outputs[0].text)
                predicted_label = pred["label"]
                human_readable_label = index2label.get(predicted_label, f"Unknown label: {predicted_label}")
            except:
                predicted_label = None
                human_readable_label = None
            
            total_tokens = len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
        
        results.append({
            "nlq": example["question"],
            "true_label": example["label"],
            "predicted_label": predicted_label,  # Numeric/original label
            "predicted_label_text": human_readable_label,  # Human-readable label
            "total_tokens": total_tokens
        })
    
    return results

def main(args):
    inputs = _load_dataset(args)
    if args.model == 'llama':
        model_args = {
            "model": "meta-llama/Llama-3.1-8B",
            "dtype": 'float16',
            "seed": 3074,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.7,
            "enforce_eager": True,
        }
        sampling_args = {
            "n": args.n_inference,  # change to 1 for greedy sampling
            "temperature": 0.7,  # change to 0 for greedy sampling
            "top_p": 0.95,
            "max_tokens": 512,
        }
        results = vllm_inference(model_args, sampling_args, inputs)
    
    elif args.model == 'gpt':
        model_args = {
            "model": args.openai_model_type if args.openai_model_type else "gpt-4o-mini"
        }
        sampling_args = {
            "n": args.n_inference,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 512,
        }
        results = openai_inference(model_args, sampling_args, inputs)
    else:
        raise NotImplementedError

    with open(args.output_path, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['llama', 'gpt'], default='gpt')
    parser.add_argument('--openai_model_type', type=str, default=None, help='e.g. gpt-4o-mini')
    parser.add_argument('--n_inference', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='output.json')
    args = parser.parse_args()
    main(args)