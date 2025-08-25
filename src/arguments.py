from dataclasses import dataclass

def get_config(exp_name):
    return getattr(ARGS, exp_name)()

@dataclass
class ARGS():
    ds: str = ''
    ds_type: str = ''
    batch_size: int = 32
    exp_path: str = 'exp1'
    database_path: str = '/lab_shared/datasets/'

    # vllm: SamplingParams
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 2048
    logprobs: int = 1
    frequency_penalty: float = 0.2

    # vllm: LLM
    model = 'meta-llama/Llama-3.2-3B-Instruct'
    max_model_len = 4096
    gpu_memory_utilization = 0.8
    enforce_eager = False
    
    @classmethod
    def tune_llm_test(cls):
        return cls(
            ds='ml', 
            ds_type='train',
            exp_path='train'
        )