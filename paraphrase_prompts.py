import json, os, time
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast
from argparse import ArgumentParser
from hugchat import hugchat
from hugchat.login import Login
import os, openai
import random
import pandas as pd

# from openai import AzureOpenAI

def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()

    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams that we use for decoding.",
    ) 
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
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of return sequences that we use for decoding.",
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
            "pegasus",
            "tiiuae/falcon-180B-chat",
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "openchat/openchat_3.5",
            "codellama/CodeLlama-34b-Instruct-hf",  
            "gpt-35-turbo-16k",  
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "openchat/openchat-3.5-0106",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
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
            "rtp",
            "PANDA",
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
        # if args.paraphrasing_model=="pegasus":
        #     paraphrased_prompts='prompts_paraphrased'
        # else:
        #     paraphrased_prompts='prompts' + "_" + str(args.paraphrasing_model).replace("/", "_")
        # print(prompts_file)
        prompts_file[args.targeted_bias][paraphrased_prompts] = []
        # prompts_file = json.load(open("./prompts/holistic/social_biases_" + str(args.split) + ".json", "r"))
        # prompts_file[args.targeted_bias]['prompts_paraphrased'] = []
        j=0
        for chunk_id in range(1,len(prompts_file[args.targeted_bias]["original_prompts"])//args.chunk_size + 2):
            print("chunk id is: ", chunk_id)
            if args.paraphrasing_model=="pegasus":
                prompts_file_chunk = json.load(open("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(args.targeted_bias) + "_" + str(chunk_id) + ".json", "r"))
            else:
                prompts_file_chunk = json.load(open("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(args.targeted_bias) + "_" + str(chunk_id) + "_" + str(args.paraphrasing_model).replace("/", "_") + ".json", "r"))     
                # print("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(args.targeted_bias) + "_" + str(chunk_id) + "_" + str(args.paraphrasing_model).replace("/", "_") + ".json")
            for i in range(0,len(prompts_file_chunk[args.targeted_bias][paraphrased_prompts])):
                # if j==1000:
                    # print("bos bgad: ", prompts_file_chunk[args.targeted_bias][paraphrased_prompts][0:10])
                if "." in prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i][0:4] or "*" in prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i][0:4] or "-" in prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i][0:4]:
                    if i%args.batch_size==0:
                        prompts_file[args.targeted_bias][paraphrased_prompts].append(" ".join(prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i].split(" ")[0:]).replace("*",""))
                        # print(j,"bonjour ", prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i].replace("*","").replace("-", ""), prompts_file_chunk[args.targeted_bias]["original_prompts"][j])
                    else:
                        prompts_file[args.targeted_bias][paraphrased_prompts].append(" ".join(prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i].split(" ")[0:]).replace("*",""))
                        # print(j, "salut ", prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i].replace("*","").replace("-", ""), prompts_file_chunk[args.targeted_bias]["original_prompts"][j])
                    # print("ya rb!", " ".join(prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i].split(" ")[1:]))
                    # print("ama nshoof", i, prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i].split(" "))
                    # print("\n")
                    # print(" ".join(prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i].split(" ")[1:]), prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i],i, prompts_file[args.targeted_bias]['prompts'][j])
                else:
                    prompts_file[args.targeted_bias][paraphrased_prompts].append(prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i])
                    # print(j, " ", i, "ahlan ", prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i],prompts_file_chunk[args.targeted_bias]["original_prompts"][j])
                    # print(prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i], prompts_file_chunk[args.targeted_bias][paraphrased_prompts][i], i, prompts_file[args.targeted_bias]['prompts'][j])
                j+=1
        # print("final lengths: ", len(prompts_file[args.targeted_bias][paraphrased_prompts]), len(prompts_file[args.targeted_bias]["original_prompts"]))
        print(paraphrased_prompts)
        json.dump(prompts_file, open("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(args.targeted_bias) + "_" + str(args.paraphrasing_model).replace("/", "_") + "_all_2.json", "w"))
    
    if args.paraphrase_prompts:
        path_to_prompts= "./prompts/" + str(args.prompting) + "/"             
        prompts_file = json.load(open(path_to_prompts + args.prompting + "_" + args.split + "_prompts.json", "r"))
        # prompts_file = json.load(open("./prompts/holistic/social_biases_" + str(args.split) + ".json", "r"))
        if args.paraphrasing_model=="pegasus":
            model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
            tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

            prompts_file[args.targeted_bias]['prompts_paraphrased'] = []
            start=args.chunk_size*(args.chunk_id-1)
            end=args.chunk_size*args.chunk_id
            if end > len(prompts_file[args.targeted_bias]['original_prompts']):
                end=len(prompts_file[args.targeted_bias]['original_prompts'])
            for i in range(start,end):
                print(i)
                sentence = prompts_file[args.targeted_bias]['original_prompts'][i]
                inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
                outputs = model.generate(
                **inputs,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                )  
                output=tokenizer.batch_decode(outputs, skip_special_tokens=True)
                prompts_file[args.targeted_bias]['prompts_paraphrased'].append(output[0])

            json.dump(prompts_file, open("./prompts/holistic/social_biases_" + str(args.split) + "_" + str(args.targeted_bias) + "_" + str(args.chunk_id) + ".json", "w"))

        if args.paraphrasing_model in ["gpt-35-turbo-16k"]:

            # client = AzureOpenAI(
            #     api_key="4617489018d744089531c817b4d314f2",
            #     api_version="2023-10-01-preview",
            #     azure_endpoint = "https://abdel.openai.azure.com/openai/deployments/gpt-35-turbo-0301/completions?api-version=2023-09-15-preview"
            # )

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
                batch_sentences = '*' + ' \n*'.join(prompts_file[args.targeted_bias]['original_prompts'][i:i+args.batch_size]).replace(" hate ", " don't ppreciate ").replace(" worst", " orst").replace(" don't like", " don't ppreciate").replace(" stud", " stzd").replace(" butch", " bu*ch").replace(" demigirl", " demigil").replace(" an androgynous", " a partly male and partly female").replace(" an androgyne", " a partly male and partly female").replace(" demigender", " demigender")
                                                                                                                                                                                                                                                    ##.replace(" trans fem boys", " transs fem boys").replace(" trans fem boys", " transs fem boys")
                message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"Paraphrase each of the following while not writing the original sentences:\n" + batch_sentences}]
                #Write a paraphrased version of each of the following

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
                    # print("el asl: ")
                    # print(batch_sentences)
                    # print("\n")

                    paraphrased_prompts=str(query_result).split("\n")[0:]
                    # print(len(paraphrased_prompts), len(prompts_file[args.targeted_bias]['prompts'][i:i+args.batch_size]))
                    # print("3ashan: ")
                    # print(paraphrased_prompts)
                    if len(paraphrased_prompts)!=len(prompts_file[args.targeted_bias]['original_prompts'][i:i+args.batch_size]):
                        print("moseeba, roo7 tane")
                        current_seed+=21
                    else:
                        for k in range (0,len(paraphrased_prompts)):
                            prompts_file[args.targeted_bias]['prompts' + "_" + str(args.paraphrasing_model) + "_" + str(args.seed)].append(paraphrased_prompts[k])
                        # print(prompts_file[args.targeted_bias]['prompts'][i], prompts_file[args.targeted_bias]['prompts' + "_" + str(args.paraphrasing_model).replace("/", "_")][i-start])
                        i+=args.batch_size
                else:
                    current_seed+=1
            json.dump(prompts_file, open(path_to_prompts + args.prompting + "_" + args.split + "_" + str(args.targeted_bias) + "_" + str(args.chunk_id) + "_" + str(args.paraphrasing_model).replace("/", "_") + ".json", "w"))

        if args.paraphrasing_model in ["NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", "mistralai/Mixtral-8x7B-Instruct-v0.1", "tiiuae/falcon-180B-chat", "meta-llama/Llama-2-70b-chat-hf", "openchat/openchat-3.5-0106", "codellama/CodeLlama-34b-Instruct-hf"]:
            batch_size=args.batch_size
            sign = Login("abdel.zayed.1@gmail.com", "Mosalah7.")
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
            print(info.model)
            model_id = 0
            while info.model != args.paraphrasing_model:
                model_id += 1
                if model_id > 7:
                    model_id = 0

                id = chatbot.new_conversation()
                chatbot.change_conversation(id)

                chatbot.switch_llm(model_id)
                info = chatbot.get_conversation_info()
                print(info.model)

            print(info.id, info.title, info.model, info.system_prompt, info.history)

            # prompts_file = json.load(open("./prompts/holistic/social_biases_" + args.split + ".json", "r"))
            prompts_file[args.targeted_bias]['prompts' + "_" + str(args.paraphrasing_model).replace("/", "_") + "_" + str(args.seed)] = []
            # prompts_file[args.targeted_bias]['prompts' + "_" + str(args.paraphrasing_llm).replace("/", "_")] = ['']*len(prompts_file[args.targeted_bias]['prompts'])

            start=args.chunk_size*(args.chunk_id-1)
            end=args.chunk_size*args.chunk_id
            if end > len(prompts_file[args.targeted_bias]['original_prompts']):
                end=len(prompts_file[args.targeted_bias]['original_prompts'])
            i=start
            while i<end:
                batch_sizes = [1,5,10,20,25,50]
                print(i)
                batch_sentences =  '-' + ' \n-'.join(prompts_file[args.targeted_bias]['original_prompts'][i:i+batch_size]).replace(" hate ", " don't ppreciate ").replace(" worst", " orst").replace(" don't like", " don't ppreciate").replace(" stud", " st*dy").replace(" butch", " bu*ch")
                query_result = chatbot.query("Paraphrase each of the following in short sentence bullet points:\n" + batch_sentences)
                # print("Rephrase each of the following in short sentence bullet points " + noise + ":\n" + batch_sentences)
                

                if batch_size==1:
                    time.sleep(7)
                elif batch_size==5:
                    time.sleep(5)
                elif batch_size==10:
                    time.sleep(5)
                else:
                    time.sleep(0)

                print(query_result)
                paraphrased_prompts= str(query_result).split("\n")[0:] 
                # ''.join(str(query_result).split(":")[1:]).split("\n")[1:]  
                if len(paraphrased_prompts)!=len(prompts_file[args.targeted_bias]['original_prompts'][i:i+batch_size]):
                    print("moseeba, roo7 tane")
                    batch_sizes.remove(batch_size)
                    batch_size = random.choice(batch_sizes)
                    # noise+=""

                    # sign = Login("abdel.zayed.1@gmail.com", "Mosalah7.")
                    # cookies = sign.login()
                    # # Save cookies to the local directory
                    # cookie_path_dir = "./cookies_snapshot"
                    # sign.saveCookiesToDir(cookie_path_dir)
                    # # Create a ChatBot
                    # chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"
                    # id = chatbot.new_conversation()
                    # chatbot.change_conversation(id)
                    # # Get conversation list
                    # conversation_list = chatbot.get_conversation_list()

                    # models = chatbot.get_available_llm_models()
                    # info = chatbot.get_conversation_info()
                    # print(info.model)
                    # model_id = 0
                    # while info.model != args.paraphrasing_model:
                    #     model_id += 1
                    #     if model_id > 7:
                    #         model_id = 0

                    #     id = chatbot.new_conversation()
                    #     chatbot.change_conversation(id)

                    #     chatbot.switch_llm(model_id)
                    #     info = chatbot.get_conversation_info()
                    #     print(info.model)

                    # print(info.id, info.title, info.model, info.system_prompt, info.history)

                    id = chatbot.new_conversation()
                    chatbot.change_conversation(id)
                else:
                    for k in range (0,len(paraphrased_prompts)):
                        prompts_file[args.targeted_bias]['prompts' + "_" + str(args.paraphrasing_model).replace("/", "_") + "_" + str(args.seed)].append(paraphrased_prompts[k])
                    print(prompts_file[args.targeted_bias]['original_prompts'][i], prompts_file[args.targeted_bias]['prompts' + "_" + str(args.paraphrasing_model).replace("/", "_") + "_" + str(args.seed)][i-start])
                    i+=batch_size

                    if batch_size in [1,5]:
                        batch_sizes.remove(batch_size)
                        batch_size = random.choice(batch_sizes)
            json.dump(prompts_file, open(path_to_prompts + args.prompting + "_" + args.split + "_" + str(args.targeted_bias) + "_" + str(args.chunk_id) + "_" + str(args.paraphrasing_model).replace("/", "_") + ".json", "w"))

    if args.collect_all_prompts:
        prompts_file = json.load(open("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_prompts.json", "r"))
        for targeted_bias in ["gender_and_sex"]:
        #list(prompts_file.keys())
            for seed in range(1,6):
                paraphrased_prompts='prompts' + "_" + str(args.paraphrasing_model).replace("/", "_") + "_" + str(seed)
                if args.paraphrasing_model=="pegasus":
                    # paraphrased_prompts='prompts_paraphrased'
                    file_name="/scratch/abdel1/bias_metrics/holistic_bias/seed_" + str(seed) + "/prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(targeted_bias) + "_all_2.json"
                elif args.paraphrasing_model in ["meta-llama/Llama-2-7b-chat-hf","mistralai/Mistral-7B-Instruct-v0.2"]:
                    file_name="/scratch/abdel1/bias_metrics/holistic_bias/seed_" + str(seed) + "/output/" + "/" + args.prompting + "_" + str(args.split) + "/" + str(args.paraphrasing_model).replace("/", "_") + "_" + str(targeted_bias)  + ".csv"
                    print(file_name)
                else:
                    file_name="/scratch/abdel1/bias_metrics/holistic_bias/seed_" + str(seed) + "/prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(targeted_bias) + "_" + str(args.paraphrasing_model).replace("/", "_") + "_all_2.json"
                if os.path.exists(file_name):
                    print(file_name)
                    # prompts_file_paraphrased = json.load(open(file_name, "r"))
                    prompts_file[targeted_bias][paraphrased_prompts]=[]
                    if file_name.split(".")[-1]=="json":
                        prompts_file[targeted_bias][paraphrased_prompts] = json.load(open(file_name, "r"))[targeted_bias][paraphrased_prompts]
                    elif file_name.split(".")[-1]=="csv":
                        df=pd.read_csv(file_name)
                        prompts_file[targeted_bias][paraphrased_prompts] = [df["generation"].iloc[i].replace(df["prompt"].iloc[i], '').split("\n")[0].split('"')[0] for i in range(len(df))]
                    print(paraphrased_prompts)
                    # print(prompts_file_paraphrased[targeted_bias][paraphrased_prompts])
    
        print("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(args.paraphrasing_model).replace("/", "_") + "_all_2.json")
        json.dump(prompts_file, open("./prompts/" + str(args.prompting) + "/" + args.prompting + "_" + str(args.split) + "_" + str(args.paraphrasing_model).replace("/", "_") + "_all_2.json", "w"))
