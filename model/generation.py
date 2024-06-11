import torch
from tqdm.notebook import tqdm
import pandas as pd
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_prompts(seed, model_name, model, tokenizer, tox_model, evaluator_honest, ppl, batch_size, 
                    max_continuation_length, max_prompt_length, output_dir, groups_and_subgroups, prompts_file, paraphrase_prompts, paraphrase_model, 
                    prompting_template, targeted_bias=None, split=None):           
    """
    Collect the model continuations and toxicity for different groups of prompts.

    Args:
        model_name: str 
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        tox_model: transformers.PreTrainedModel
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
                if(paraphrase_prompts):
                    prompts = list(map(lambda x:'"' + x + '" can be paraphrased as "', prompts))
                    print("My prompts:", prompts)
                generations, toxicity_scores = gen_prompt(
                    seed, model_name, model, tokenizer, prompts, tox_model, batch_size, max_continuation_length, max_prompt_length
                )
                domain_results.extend(
                    [
                        {
                            "group": group, 
                            "title": title,
                            "prompt": prompt_text,
                            "generation": gen,
                            "toxicity_score": tox_score,
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
        # This is only for the HONEST metric
        honest_results=pd.DataFrame()  
        for title in full_results_pd["title"].unique():
            honest_score=evaluator_honest.honest_dataframe(list(full_results_pd[full_results_pd["title"]==title]["generation"]), prompting_template)
            honest_score[1]["honest_score"]=honest_score[0]
            honest_results=pd.concat([honest_results, honest_score[1]], ignore_index=True)

        full_results_pd=pd.concat([full_results_pd, honest_results], axis=1)
    
    full_results_pd.to_csv(output_dir + model_name + "_" + targeted_bias + "_" + paraphrase_model + "_reproducibility.csv",index=False,)
    
    
def gen_prompt(
    seed, model_name, model, tokenizer, data, tox_model, batch_size, max_continuation_length, max_prompt_length
):
    """
    Given some prompts, generate model continuation and measure toxicity.
    Args:
        model_name: str
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        data: list[str]
        tox_model: transformers.PreTrainedModel
        batch_size: int
        max_continuation_length: int
        max_prompt_length: int

    Returns:
        outputs: list[str]
        toxicity_scores: list[float]
    """
    outputs, toxicity_scores = [], []
    with torch.no_grad():
        for idx in tqdm(range(0, len(data), batch_size)):
            batch = data[idx : idx + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length)
            print("idx is ", idx) 
            output_sequences = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                max_length=len(inputs["input_ids"][0]) + max_continuation_length,
                do_sample=True,
            )
            decoded_sequences = tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True
            )
            print(decoded_sequences)
            # toxicity_scores += tox_model.predict([x.replace("\n", " ").lower() for x in decoded_sequences])["toxicity"]
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
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl[data + "_" + split] = round(float(torch.exp(torch.stack(nlls).mean())), 3)

    return ppl

