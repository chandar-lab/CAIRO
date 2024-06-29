import torch
from transformers import LlamaForCausalLM, AutoModelWithLMHead, AutoTokenizer, GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoXForCausalLM, GPTNeoForCausalLM, GPTNeoXTokenizerFast, BloomForCausalLM, AutoTokenizer, AutoModelWithLMHead, BloomTokenizerFast, OPTForCausalLM, GPT2Tokenizer, GPTJForCausalLM, AutoTokenizer, BloomTokenizerFast, GPT2Tokenizer, BloomForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM, OPTForCausalLM, GPTJForCausalLM, AutoModelForCausalLM, AutoModelWithLMHead
# This file used to download the models from huggingface and save them in the cached_models folder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_name in ["EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B","gpt2","EleutherAI/gpt-j-6B","distilgpt2","EleutherAI/gpt-neo-125M","EleutherAI/pythia-160m","EleutherAI/pythia-410m","EleutherAI/pythia-1b", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b"]:
    print(model_name)
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2",  "gpt2-xl"]:
        model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")   # Initialize tokenizer
        
    elif model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("saved_models/cached_tokenizers/" + model_name, padding_side="left")

    elif model_name in ["EleutherAI/gpt-j-6B"]:
        model =  GPTJForCausalLM.from_pretrained(model_name,revision="float16", torch_dtype=torch.float16,).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")

    elif model_name in ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]:
        model = OPTForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    elif model_name in ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m","EleutherAI/pythia-1b","EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b","EleutherAI/pythia-6.9b","EleutherAI/pythia-12b"]:
        model = GPTNeoXForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    elif model_name in ["meta-llama/Llama-2-7b-chat-hf"]:
        print("./saved_models/cached_models/" + model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = LlamaForCausalLM.from_pretrained(model_name, token= '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').to(device)

    elif model_name in ['mistralai/Mistral-7B-Instruct-v0.2']:
        print("./saved_models/cached_models/" + model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
        ).to(device)

    model.save_pretrained("./saved_models/cached_models/" + model_name)
    tokenizer.save_pretrained("./saved_models/cached_tokenizers/" + model_name)
    print("./saved_models/cached_models/" + model_name)
