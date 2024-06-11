import json, os
from argparse import ArgumentParser
import os, openai
import pandas as pd

def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="The number of examples that we paraphrase at once.",
    ) 
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=1,
        help="The index of the chunk that we are paraphrasing.",
    ) 
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="The batch size used while paraphrasing.",
    )  
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="The seed that we are using. We normally run every experiment for 5 seeds.",
    )  
    parser.add_argument(
        "--collect_paraphrased_chunks",
        type=bool,
        default=False,
        help="Whether or not to collect all paraphrased chunks into one file.",
    )
    parser.add_argument(
        "--paraphrase_prompts",
        type=bool,
        default=False,
        help="Whether or not to do paraphrasing.",
    )
    parser.add_argument(
        "--paraphrasing_model",
        choices=[
            "meta-llama/Llama-2-7b-chat-hf",
            "gpt-35-turbo-16k",  
            "mistralai/Mistral-7B-Instruct-v0.2",
            "openchat/openchat-3.5-0106",
            None,
        ],
        default="gpt-35-turbo-16k",
        help="The llm used for paraphrasing",
    )     
    parser.add_argument(
        "--collect_all_prompts",
        type=bool,
        default=False,
        help="Whether or not to collect all bias prompts into one file",
    )
    parser.add_argument(
        "--split",
        choices=[
            "valid",
            "test",       
        ],
        default="valid",
        help="The split of the prompts that we are parphrasing.",
    )   
    parser.add_argument("-", "--model_list", nargs="+", default=[])
    parser.add_argument(
        "--prompting",
        choices=[
            "BOLD",
            "holistic",
            "HONEST",
        ],
        default="BOLD",
        help="Type of prompt used for the language model",
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
        ],
        default="gender_and_sex",
        help="The group for which biased is assessed",
    )    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.collect_paraphrased_chunks:
        prompts_file = json.load(open("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_prompts.json", "r"))
        paraphrased_prompts='prompts' + "_" + str(args.paraphrasing_model).replace("/", "_") + "_" + str(args.seed)
        prompts_file[args.targeted_bias][paraphrased_prompts] = []
        j=0
        for chunk_id in range(1,len(prompts_file[args.targeted_bias]["original_prompts"])//args.chunk_size + 2):
            print("chunk id is: ", chunk_id)
            prompts_file_chunk = json.load(open("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(args.targeted_bias) + "_" + str(chunk_id) + "_" + str(args.paraphrasing_model).replace("/", "_") + ".json", "r"))     
            for i in range(0,len(prompts_file_chunk[args.targeted_bias][paraphrased_prompts])):
                if "." in prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i][0:4] or "*" in prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i][0:4] or "-" in prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i][0:4]:
                    if i%args.batch_size==0:
                        prompts_file[args.targeted_bias][paraphrased_prompts].append(" ".join(prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i].split(" ")[0:]).replace("*",""))
                    else:
                        prompts_file[args.targeted_bias][paraphrased_prompts].append(" ".join(prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i].split(" ")[0:]).replace("*",""))
                else:
                    prompts_file[args.targeted_bias][paraphrased_prompts].append(prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i])
                j+=1
        print(paraphrased_prompts)
        json.dump(prompts_file, open("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(args.targeted_bias) + "_" + str(args.paraphrasing_model).replace("/", "_") + "_all_2.json", "w"))
    
    if args.paraphrase_prompts:
        path_to_prompts= "./prompts/" + str(args.prompting) + "/"             
        prompts_file = json.load(open(path_to_prompts + args.prompting + "_" + args.split + "_prompts.json", "r"))
        if args.paraphrasing_model in ["gpt-35-turbo-16k"]:
            openai.api_type = "azure"
            openai.api_base = "https://abdel.openai.azure.com/"
            openai.api_version = "2023-09-15-preview"
            openai.api_key = "4617489018d744089531c817b4d314f2"
            openai.azure_endpoint = "https://abdel.openai.azure.com/openai/deployments/gpt-35-turbo-0301/completions?api-version=2023-09-15-preview"
            
            prompts_file[args.targeted_bias]['prompts' + "_" + str(args.paraphrasing_model) + "_" + str(args.seed)] = []
            start=args.chunk_size*(args.chunk_id-1)
            end=args.chunk_size*args.chunk_id
            if end > len(prompts_file[args.targeted_bias]['original_prompts']):
                end=len(prompts_file[args.targeted_bias]['original_prompts'])
            i=start
            current_seed=args.seed*1000 # we increment it when the model fails to answer
            while i<end:
                batch_sentences = '*' + ' \n*'.join(prompts_file[args.targeted_bias]['original_prompts'][i:i+args.batch_size]).replace(" hate ", " don't ppreciate ").replace(" worst", " orst").replace(" don't like", " don't ppreciate").replace(" stud", " stzd").replace(" butch", " bu*ch").replace(" demigirl", " demigil").replace(" an androgynous", " a partly male and partly female").replace(" an androgyne", " a partly male and partly female").replace(" demigender", " demigender")                                                                                                                                                                                                                                                   ##.replace(" trans fem boys", " transs fem boys").replace(" trans fem boys", " transs fem boys")
                message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"Paraphrase each of the following while not writing the original sentences:\n" + batch_sentences}]
                print(batch_sentences)
                completion = openai.ChatCompletion.create(
                engine=args.paraphrasing_model,
                messages = message_text,
                temperature=0.95,
                max_tokens=800,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                seed=current_seed,
                stop=None
                )
                # time.sleep(20)
                print(i)
                if len(completion.choices[0].message)!=1:
                    # this is to make sure that the paraphrasing is not empty
                    query_result = completion.choices[0].message.content
                    print(query_result)
                    print("\n")
                    paraphrased_prompts=str(query_result).split("\n")[0:]
                    if len(paraphrased_prompts)!=len(prompts_file[args.targeted_bias]['original_prompts'][i:i+args.batch_size]):
                        print("moseeba, roo7 tane")
                        current_seed+=21
                    else:
                        for k in range (0,len(paraphrased_prompts)):
                            prompts_file[args.targeted_bias]['prompts' + "_" + str(args.paraphrasing_model) + "_" + str(args.seed)].append(paraphrased_prompts[k])
                        i+=args.batch_size
                else:
                    current_seed+=1
            json.dump(prompts_file, open(path_to_prompts + args.prompting + "_" + args.split + "_" + str(args.targeted_bias) + "_" + str(args.chunk_id) + "_" + str(args.paraphrasing_model).replace("/", "_") + "_reproducibility.json", "w"))

    if args.collect_all_prompts:
        prompts_file = json.load(open("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_prompts.json", "r"))
        for targeted_bias in ["gender_and_sex"]:
            for seed in range(1,6):
                paraphrased_prompts='prompts' + "_" + str(args.paraphrasing_model).replace("/", "_") + "_" + str(seed)
                if args.paraphrasing_model in ["meta-llama/Llama-2-7b-chat-hf","mistralai/Mistral-7B-Instruct-v0.2"]:
                    file_name="/scratch/abdel1/bias_metrics/holistic_bias/seed_" + str(seed) + "/output/" + "/" + args.prompting + "_" + str(args.split) + "/" + str(args.paraphrasing_model).replace("/", "_") + "_" + str(targeted_bias)  + ".csv"
                    print(file_name)
                else:
                    file_name="/scratch/abdel1/bias_metrics/holistic_bias/seed_" + str(seed) + "/prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(targeted_bias) + "_" + str(args.paraphrasing_model).replace("/", "_") + "_all_2.json"
                if os.path.exists(file_name):
                    print(file_name)
                    prompts_file[targeted_bias][paraphrased_prompts]=[]
                    if file_name.split(".")[-1]=="json":
                        prompts_file[targeted_bias][paraphrased_prompts] = json.load(open(file_name, "r"))[targeted_bias][paraphrased_prompts]
                    elif file_name.split(".")[-1]=="csv":
                        df=pd.read_csv(file_name)
                        prompts_file[targeted_bias][paraphrased_prompts] = [df["generation"].iloc[i].replace(df["prompt"].iloc[i], '').split("\n")[0].split('"')[0] for i in range(len(df))]
                    print(paraphrased_prompts)
    
        print("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(args.paraphrasing_model).replace("/", "_") + "_all_2.json")
        json.dump(prompts_file, open("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(args.paraphrasing_model).replace("/", "_") + "_all_2.json", "w"))
