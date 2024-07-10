import torch
from transformers import AutoTokenizer, BloomTokenizerFast, BloomForCausalLM, GPT2Tokenizer
from transformers import AutoModelForCausalLM, GPTNeoForCausalLM, AutoModelForCausalLM, AutoModelWithLMHead
from transformers import LlamaForCausalLM, GPTNeoXForCausalLM, OPTForCausalLM, GPTJForCausalLM

def load_model_and_tokenizer(model_name, device):
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2",  "gpt2-xl"]:
        model = AutoModelWithLMHead.from_pretrained("./saved_models/cached_models/" + model_name).to(device)

    elif model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        model = GPTNeoForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name).to(device)

    elif model_name in ["EleutherAI/gpt-j-6B"]:
        model = GPTJForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name).to(device)

    elif model_name in ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]:
        model = OPTForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name).to(device)

    elif model_name in ["bigscience/bloom-560m", "bigscience/bloom-1b1","bigscience/bloom-3b", "bigscience/bloom-7b1"]:
        model = BloomForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name).to(device)

    elif model_name in ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m","EleutherAI/pythia-1b","EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b","EleutherAI/pythia-6.9b","EleutherAI/pythia-12b"]:
        model = GPTNeoXForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name).to(device)

    elif model_name in ["meta-llama/Llama-2-7b-chat-hf"]:
        model = LlamaForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name, device_map="auto", load_in_4bit=True, revision="float16",torch_dtype=torch.float16,
                                                token= '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').bfloat16()

    elif model_name in ['mistralai/Mistral-7B-Instruct-v0.2']:
        model = AutoModelForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name, device_map="auto", load_in_4bit=True, revision="float16",torch_dtype=torch.float16).bfloat16()

    if model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        tokenizer = GPT2Tokenizer.from_pretrained("./saved_models/cached_tokenizers/" + model_name, padding_side="left") # Initialize tokenizer for generation to the left
    
    elif model_name in ["bigscience/bloom-560m", "bigscience/bloom-1b1","bigscience/bloom-3b", "bigscience/bloom-7b1"]:
        tokenizer = BloomTokenizerFast.from_pretrained("./saved_models/cached_tokenizers/" + model_name, padding_side="left") 

    elif model_name in ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b","EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m","EleutherAI/pythia-1b","EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b","EleutherAI/pythia-6.9b","EleutherAI/pythia-12b", "gpt2","gpt2-medium","gpt2-large","gpt2-xl","distilgpt2", "EleutherAI/gpt-j-6B"]:
        tokenizer = AutoTokenizer.from_pretrained("./saved_models/cached_tokenizers/" + model_name, load_in_4bit=True, padding_side="left")

    return model, tokenizer

