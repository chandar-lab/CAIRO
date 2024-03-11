import torch
from tqdm.notebook import tqdm
import pandas as pd
import time
# import openai
import json
# from openai import AzureOpenAI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_prompts(seed, model_name, model, tokenizer, tox_model, sentiment_analyzer, evaluator_honest, ppl, batch_size, 
                    max_continuation_length, max_prompt_length, output_dir, groups_and_subgroups, prompts_file, paraphrase_prompts, paraphrase_model, 
                    prompting_template, targeted_bias=None, split=None):           
    """
    Collect the model continuations, toxicity, and sentiment for different groups of prompts.

    Args:
        model_name: str 
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        tox_model: transformers.PreTrainedModel
        sentiment_analyzer: transformers.PreTrainedModel
        ppl: dict[str:float]
        batch_size: int
        max_continuation_length: int
        max_prompt_length: int
        paraphrase_model: str
        paraphrase_prompts: bool
        output_dir: str
        groups: list[str]
        prompts_file: dict[str:dict[str:list[str]]]
        targeted_bias: str
        split: str

    Returns:
        None
    """

    domain_results = []
    full_results = []
    for group, group_prompts in tqdm(prompts_file.items()):
        for title, prompts in group_prompts.items():
                print(title)
                if (group != targeted_bias) and (targeted_bias != None):
                    continue
                if (str(paraphrase_model) != title) and (paraphrase_model!=None) :
                    # these are the cases where the user wants to paraphrase the prompts using a specific model
                    continue
                # title = title.replace("_", " ").lower()
                if(paraphrase_prompts):
                    prompts = list(map(lambda x:'"' + x + '" can be paraphrased as "', prompts))
                    print("My prompts:", prompts)
                generations, toxicity_scores = gen_prompt(
                    seed, model_name, model, tokenizer, prompts, tox_model, sentiment_analyzer, batch_size, max_continuation_length, max_prompt_length
                )
                # prompt_types=["original"]*len(prompts)
                domain_results.extend(
                    [
                        {
                            "group": group, 
                            "title": title,
                            "prompt": prompt_text,
                            "generation": gen,
                            "toxicity_score": tox_score,
                            # "sentiment_score": sent_score,
                            "Perplexity wikitext": ppl["wikitext" + "_" + split],
                            "Perplexity prompts": ppl[title + "_" + split],
                            "split": split,
                        }
                        # for gen, prompt_text in zip(
                        #     generations, prompts
                        # )
                        for gen, prompt_text, tox_score in zip(
                            generations, prompts, toxicity_scores
                        )
                    ]
                )

    full_results.extend(domain_results)
    full_results_pd = pd.DataFrame(full_results)        
    # this is to add the subgroups to the results
    num_paraphrases=int(len(full_results_pd)/len(list(groups_and_subgroups["subgroups"][groups_and_subgroups["groups"] == targeted_bias])))
    full_results_pd["subgroup"] = (list(groups_and_subgroups["subgroups"][groups_and_subgroups["groups"] == targeted_bias])*num_paraphrases)[0:len(full_results_pd)]



    if prompting_template!=None:
        # This is for the HONEST metric
        honest_results=pd.DataFrame()  
        for title in full_results_pd["title"].unique():
            honest_score=evaluator_honest.honest_dataframe(list(full_results_pd[full_results_pd["title"]==title]["generation"]), prompting_template)
            honest_score[1]["honest_score"]=honest_score[0]
            honest_results=pd.concat([honest_results, honest_score[1]], ignore_index=True)

        full_results_pd=pd.concat([full_results_pd, honest_results], axis=1)
    
    full_results_pd.to_csv(output_dir + model_name + "_" + targeted_bias + "_" + paraphrase_model + ".csv",index=False,)
    
    
def gen_prompt(
    seed, model_name, model, tokenizer, data, tox_model, sentiment_analyzer, batch_size, max_continuation_length, max_prompt_length
):
    """
    Given some prompts, generate model continuation and measure both toxicity and sentiment scores.

    Args:
        model_name: str
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        data: list[str]
        tox_model: transformers.PreTrainedModel
        sentiment_analyzer: transformers.PreTrainedModel
        batch_size: int
        max_continuation_length: int
        max_prompt_length: int

    Returns:
        outputs: list[str]
        toxicity_scores: list[float]
        sentiment_scores: list[float]
    """
    outputs, toxicity_scores = [], []

    i=0
    if model_name in ["mistralai_Mistral-7B-Instruct-v0.1", "tiiuae_falcon-180B-chat", "meta-llama_Llama-2-70b-chat-hf", "openchat_openchat_3.5", "codellama_CodeLlama-34b-Instruct-hf"]:  
        # print("shoof de ", len(data), i, model_name, batch_size)
        while i<len(data):
                print(i)
                # print(' \n '.join(data[i:i+batch_size]))
                query_result = model.query("Add one more sentence to each of the following in separate lines: \n" + ' \n '.join(data[i:i+batch_size]))
                time.sleep(0.5)
                # print(len(paraphrased_prompts), len(data[i:i+batch_size]))
                paraphrased_prompts=str(query_result).split("\n")[0:]
                if len(paraphrased_prompts)!=len(data[i:i+batch_size]):
                    print("moseeba, roo7 tane")
                    id = model.new_conversation()
                    model.change_conversation(id)
                else:
                    print("mabrook")
                    i+=batch_size
                    toxicity_scores += tox_model.predict([x.replace("\n", " ").lower() for x in paraphrased_prompts])["toxicity"]
                    outputs += paraphrased_prompts
                    break

    elif model_name in ["gpt-35-turbo-16k"]:

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

        current_seed=seed*1000 # we increment it when the model fails to answer
        while i<len(data):
            message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"Rewrite the following while adding one sentence to each:\n" + '*' + '\n*'.join(data[i:i+batch_size])}]

            completion = openai.ChatCompletion.create(
            engine=model_name,
            messages = message_text,
            temperature=0.95,
            max_tokens=800,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            seed=current_seed,
            stop=None
            )


            print(i)
            if len(completion.choices[0].message)!=1:
                # this is to make sure that the paraphrasing is not empty
                query_result = completion.choices[0].message.content
                time.sleep(25)
                
                print(query_result)
                print("\n")
                # print("el asl: ")
                # print(batch_sentences)
                # print("\n")

                paraphrased_prompts=str(query_result).split("\n")[0:]
                # print(len(paraphrased_prompts), len(prompts_file[args.targeted_bias]['prompts'][i:i+args.batch_size]))
                # print("3ashan: ")
                # print(paraphrased_prompts)
                if len(paraphrased_prompts)!=len(data[i:i+batch_size]):
                    print("moseeba, roo7 tane")
                    current_seed+=1
                else:
                    print("mabrook")
                    i+=batch_size
                    toxicity_scores += tox_model.predict([x.replace("\n", " ").lower() for x in paraphrased_prompts])["toxicity"]
                    outputs += paraphrased_prompts

                    # for k in range (0,len(paraphrased_prompts)):
                    #     prompts_file[args.targeted_bias]['prompts' + "_" + str(args.paraphrasing_model) + "_" + str(args.seed)].append(paraphrased_prompts[k])
                    # # print(prompts_file[args.targeted_bias]['prompts'][i], prompts_file[args.targeted_bias]['prompts' + "_" + str(args.paraphrasing_model).replace("/", "_")][i-start])
                    # i+=args.batch_size
            else:
                current_seed+=1

        # while i<len(data):
        #         print(i)
        #         # print(' \n '.join(data[i:i+batch_size]))

        #         # query_result = openai.completions.create(
        #         # model=model_name,
        #         # prompt="Paraphrase the following: \n" + '\n'.join(data[i:i+batch_size]),
        #         # temperature=0.95,
        #         # max_tokens=800,
        #         # top_p=1,
        #         # frequency_penalty=0,
        #         # presence_penalty=0,
        #         # stop=None)
        #         # # print('*' + '\n*'.join(data[i:i+batch_size]))

        #         # time.sleep(0.1)
        #         #''.join(data[i:i+batch_size])+
        #         paraphrased_prompts=str(query_result.choices[0].text)
        #         #.split('\n')[0:]
        #         print(paraphrased_prompts)
        #         if len(paraphrased_prompts)!=len(data[i:i+batch_size]):
        #             print("moseeba, roo7 tane")
        #         else:
        #             print("mabrook")
        #             i+=batch_size
        #         toxicity_scores += tox_model.predict([x.replace("\n", " ").lower() for x in paraphrased_prompts])["toxicity"]
        #         outputs += paraphrased_prompts
        #             # break
    else:    
        for idx in tqdm(range(0, len(data), batch_size)):

            batch = data[idx : idx + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length)
            print("idx is ", idx) 

            output_sequences = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                max_length=len(inputs["input_ids"][0]) + max_continuation_length,
                do_sample=True,
                # stop='\n',
            )
            decoded_sequences = tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True
            )
            print(decoded_sequences)
            toxicity_scores += tox_model.predict([x.replace("\n", " ").lower() for x in decoded_sequences])["toxicity"]
            # sentiment_scores.append(sentiment_analyzer.polarity_scores([x.replace("\n", " ").lower() for x in decoded_sequences])["compound"])
            outputs += decoded_sequences
            # break
        
    return outputs, toxicity_scores


def compute_ppl(model, tokenizer, stride, max_position_embeddings, prompting, targeted_bias, split):
    """
    Compute perplexity of the model. Copied from https://huggingface.co/docs/transformers/perplexity

    Args:
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        stride: int
        max_position_embeddings: int
        prompting: str
        targeted_bias: str
        split: str
    
    Returns:
        ppl: float
    """
    ppl = {}
    # print("./prompts/" + str(prompting) + "/" + prompting + "_" + split + "_prompts.json")
    # x=json.load(open("./prompts/" + str(prompting) + "/" + prompting + "_" + split + "_prompts.json", "r"))
    # print("abdel")
    for data in list(json.load(open("./prompts/" + str(prompting) + "/" + prompting + "_" + split + "_prompts.json", "r"))[targeted_bias].keys()) + ["wikitext"]:
        # we compute the perplexity of both the wikitext (to measure the language modeling ability of the model) and the prompts (to measure the overlap between the prompts and the model's training data)
        print(data)
        if data == "wikitext":
            names = []    
            with open("./model/wikitext-2-raw-v1_" + split + ".txt", 'r') as fp:
                for line in fp:
                    # remove linebreak from a current name
                    # linebreak is the last character of each line
                    x = line

                    # add current item to the list
                    names.append(x)
        else:
            names = json.load(open("./prompts/" + str(prompting) + "/" + prompting + "_" + split + "_prompts.json", "r"))[targeted_bias][data]

        encodings = tokenizer("".join(names) , return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_position_embeddings, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl[data + "_" + split] = round(float(torch.exp(torch.stack(nlls).mean())), 3)
        # ppl[data + "_" + split] = 0

    return ppl

