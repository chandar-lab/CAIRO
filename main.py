import torch
import wandb
import json
from argparse import ArgumentParser
from pathlib import Path
from model.generation import process_prompts,compute_ppl
from utils import parameters_to_prune
#, LlamaForCausalLM, LlamaTokenizerFast
#transformers_pruning_new
#transformers_pruning
# from transformers import LlamaForCausalLM, LlamaTokenizerFast
from transformers import AutoTokenizer, BloomTokenizerFast, GPT2Tokenizer
from transformers import AutoModelForCausalLM, BloomForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM, OPTForCausalLM, GPTJForCausalLM, AutoModelForCausalLM, AutoModelWithLMHead
from hugchat import hugchat
from hugchat.login import Login
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from honest import honest
# from transformers import LlamaTokenizer, MixtralForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="The seed that we are using. We normally run every experiment for 5 seeds.",
    )                          
    parser.add_argument(
        "--model",
        choices=[
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "distilgpt2",
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
            "EleutherAI/gpt-j-6B",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "bigscience/bloom-560m",
            "bigscience/bloom-1b1",
            "bigscience/bloom-3b",
            "bigscience/bloom-7b1",
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b",
            "EleutherAI/pythia-12b",     
            "meta-llama/Llama-2-7b",   
            "meta-llama/Llama-2-7b-chat-hf",    
            "tiiuae/falcon-180B-chat", 
            "meta-llama/Llama-2-70b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "openchat/openchat_3.5",
            "tiiuae/falcon-7b-instruct",
            "gpt-35-turbo-16k"
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "mistralai/Mistral-7B-Instruct-v0.2",
        ],
        default="EleutherAI/gpt-neo-125M",
        help="Type of language generation model used",
    )    
    parser.add_argument("-paraphrasing_model", "--paraphrasing_model_list", nargs="+", default=[])
    # parser.add_argument(
    #     "--paraphrasing_model",
    #     choices=[
    #         "gpt-35-turbo-16k",
    #         "Mixtral-8x7B-Instruct-v0.1",
    #         "original",
    #         None,
    #     ],
    #     default=None,
    #     help="Type of prompt used for the language model. When not specified (i.e. None), we go over all the versions of paraphrasing.",
    # )          
    parser.add_argument(
        "--prompting",
        choices=[
            "rtp",
            "PANDA",
            "BOLD",
            "holistic",
            "HONEST",
        ],
        default="holistic",
        help="Type of prompt used for the language model",
    )   
    parser.add_argument(
        "--targeted_group_PANDA",
        choices=[
            "woman",
            "non-binary",
            "asian",
            "black",
            "man",
            "middle-aged",
            "white",
            "adult",
            "hispanic",
            "young",
            "child",
            "pacific-islander",
            "native-american",
            "senior",            
        ],
        default=None,
        help="The group for which biased is assessed using the PANDA framework",
    )         
    parser.add_argument(
        "--targeted_bias",
        choices=[
            "characteristics",
            "ability",
            "gender_and_sex",
            "socioeconomic_class",
            "race_ethnicity",  
            "body_type",
            "cultural",
            "religion",
            "age",
            "nonce",
            "sexual_orientation",  
            "political_ideologies",  
            "nationality",
            "NaN",       
            "gender",
            "political_ideology",
            "profession",
            "race",
            "religious_ideology",  
            "female", 
            "male", 
            "queer_gender_pronoun", 
            "queer_gender", 
            "queer_gender_xenogender",
            "queer",
            "queer_orientation",
            "nonqueer_gender",
            "nonqueer",
            "nonqueer_orientation",
            None,
        ],
        default=None,
        help="The group for which biased is assessed. When not specified (i.e. None), we go over all the biases.",
    )            
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for the language model.",
    )
    #128
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride used for computing the model perplexity. This corresponds to the number of tokens the model conditions on at each step.",
    ) 
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="The maximum number of tokens that the model can condition on while generating the probability distribution over the next token.",
    )    
    parser.add_argument(
        "--max_continuation_length",
        type=int,
        default=25,
        help="The maximum length of the continuation for the language generation model",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=100,
        help="The maximum length of the prompt for the language generation model",
    )    
    parser.add_argument(
        "--output_dir",
        default="output/",
        help="Directory to the output",
    )
    parser.add_argument(
        "--paraphrase_prompts",
        type=bool,
        default=False,
        help="Whether or not to paraphrase the prompts.",
    )                   
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)

    path_to_prompts= "./prompts/" + str(args.prompting) + "/"

    tokenizer, model = None, None # these models are too big to be saved and loaded, so we use api calls to interact with them

    if args.model in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2",  "gpt2-xl"]:
        model = AutoModelWithLMHead.from_pretrained("./saved_models/cached_models/" + args.model).to(device)

    elif args.model in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        model = GPTNeoForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model).to(device)

    elif args.model in ["EleutherAI/gpt-j-6B"]:
        model = GPTJForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model).to(device)
        #revision="float16",torch_dtype=torch.float16,

    elif args.model in ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]:
        model = OPTForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model).to(device)

    elif args.model in ["bigscience/bloom-560m", "bigscience/bloom-1b1","bigscience/bloom-3b", "bigscience/bloom-7b1"]:
        model = BloomForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model).to(device)

    elif args.model in ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m","EleutherAI/pythia-1b","EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b","EleutherAI/pythia-6.9b","EleutherAI/pythia-12b"]:
        model = GPTNeoXForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model).to(device)

    elif args.model in ["meta-llama/Llama-2-7b-chat-hf"]:
        model = LlamaForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model, device_map="auto", load_in_4bit=True, revision="float16",torch_dtype=torch.float16,
                                                token= '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').bfloat16()

    elif args.model in ['mistralai/Mistral-7B-Instruct-v0.2']:
        model = AutoModelForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model, device_map="auto", load_in_4bit=True, revision="float16",torch_dtype=torch.float16).bfloat16()

    elif args.model in ["Nous-Hermes-2-Mixtral-8x7B-DPO"]:
        model = MixtralForCausalLM.from_pretrained(
            "./saved_models/cached_models/" + args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=False,
            load_in_4bit=True,
            use_flash_attention_2=True
        ).to(device)

    elif args.model in ["tiiuae/falcon-7b-instruct"]:   
        model = AutoModelForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model,trust_remote_code=True).half().to(device)
#load_in_4bit=True,
    elif args.model in ["mistralai/Mistral-7B-Instruct-v0.1", "tiiuae/falcon-180B-chat", "meta-llama/Llama-2-70b-chat-hf", "openchat/openchat_3.5", "codellama/CodeLlama-34b-Instruct-hf"]:
        print((args.model + "_" + args.targeted_holistic_bias + "_" + str(args.seed) + "@sharklasers.com").replace("/",""))
        sign = Login((args.model + "_" + args.targeted_holistic_bias + "_" + str(args.seed) + "@sharklasers.com").replace("/",""), "Thepianist7.")
        cookies = sign.login()
        # Save cookies to the local directory    
        cookie_path_dir = "./cookies_snapshot"
        sign.saveCookiesToDir(cookie_path_dir)
        # Create a ChatBot
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"
        id = chatbot.new_conversation()
        chatbot.change_conversation(id)
        # Get conversation list
        conversation_list = chatbot.get_conversation_list()

        # Get the available models (not hardcore)
        models = chatbot.get_available_llm_models()
        info = chatbot.get_conversation_info()
        model_id = 0
        while info.model != args.model:
            model_id += 1
            if model_id > 4:
                model_id = 0

            id = chatbot.new_conversation()
            chatbot.change_conversation(id)

            chatbot.switch_llm(model_id)
            info = chatbot.get_conversation_info()
            print(info.model)

        print(info.id, info.title, info.model, info.system_prompt, info.history)
        model=chatbot

    if args.model in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        tokenizer = GPT2Tokenizer.from_pretrained("./saved_models/cached_tokenizers/" + args.model, padding_side="left") # Initialize tokenizer for generation to the left
    
    elif args.model in ["bigscience/bloom-560m", "bigscience/bloom-1b1","bigscience/bloom-3b", "bigscience/bloom-7b1"]:
        tokenizer = BloomTokenizerFast.from_pretrained("./saved_models/cached_tokenizers/" + args.model, padding_side="left") 

    elif args.model in ['mistralai/Mistral-7B-Instruct-v0.2', "meta-llama/Llama-2-7b-chat-hf","tiiuae/falcon-7b-instruct", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b","EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m","EleutherAI/pythia-1b","EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b","EleutherAI/pythia-6.9b","EleutherAI/pythia-12b", "gpt2","gpt2-medium","gpt2-large","gpt2-xl","distilgpt2", "EleutherAI/gpt-j-6B"]:
        tokenizer = AutoTokenizer.from_pretrained("./saved_models/cached_tokenizers/" + args.model, load_in_4bit=True, padding_side="left")

    elif args.model in ["Nous-Hermes-2-Mixtral-8x7B-DPO"]:
        tokenizer = LlamaTokenizer.from_pretrained("./saved_models/cached_tokenizers/" + args.model, trust_remote_code=True)

    splits = ["valid","test"]
    # splits = ["valid"]
    #, "test"

    if tokenizer!= None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    
    # ppl = {}
    # ppl["valid"], ppl["test"] = 0,0        

    tox_model = torch.load("./saved_models/unbiased/unbiased.pt")
    tox_model.device = device 
    # tox_model = None
    model_name = args.model.replace("/", "_")
    sentiment_analyzer = SentimentIntensityAnalyzer() 
    evaluator_honest = honest.HonestEvaluator("en")

    for split in splits:
        if args.model!="gpt-35-turbo-16k":
          ppl = compute_ppl(model,tokenizer, args.stride, args.max_length, args.prompting, args.targeted_bias, split)
          print(ppl)
        else:
          ppl={}
          ppl["prompts_" + split], ppl["wikitext_" + split] = 0,0 
        output_dir = args.output_dir + "/" + str(args.prompting)


        groups_and_subgroups = {}
        groups_and_subgroups["groups"] = {}
        groups_and_subgroups["subgroups"] = {}
        groups_and_subgroups["groups"] =  json.load(open(path_to_prompts + args.prompting + "_" + split + "_groups.json", "r"))["groups"]
        groups_and_subgroups["subgroups"] = json.load(open(path_to_prompts + args.prompting + "_" + split + "_groups.json", "r"))["subgroups"]
        groups_and_subgroups = pd.DataFrame.from_dict(groups_and_subgroups)

        prompts_file = json.load(open(path_to_prompts + args.prompting + "_" + split + "_prompts.json", "r"))

        if args.prompting == "HONEST":
            prompting_template=json.load(open(path_to_prompts + args.prompting + "_" + split + "_templates.json", "r"))
        else:
            prompting_template = None
            # this files has the information about the groups and subgroups that are targeted bias (for example, different religions, genders, etc.)
        
        output_dir += "_" + split + "/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for paraphrasing_model in args.paraphrasing_model_list:
            process_prompts(args.seed, model_name, model, tokenizer, tox_model, sentiment_analyzer, evaluator_honest, ppl, args.batch_size, args.max_continuation_length, args.max_prompt_length, output_dir, groups_and_subgroups, prompts_file, args.paraphrase_prompts, paraphrasing_model, prompting_template, args.targeted_bias, split)
        



            
        






        




            
