import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
import torch
from argparse import ArgumentParser
from random import choices
import time
import itertools
# this file is used to collect the results from the different runs of the experiments and save them in a csv file

def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--paraphrasing_model",
        choices=[
            "Chatgpt",
            "Mistral",
            "Llama",
            "Chatgpt_Llama_Mistral",
        ],
        default="Chatgpt_Llama_Mistral",
        help="The model used to generate the paraphrases. If not specified, it ",
    ) 
    parser.add_argument(
        "--experiment",
        choices=[
            "robustness_effect",
            "data_augmentation_effect",
            "collect_all_csv_files",
        ],
        default="robustness_effect",
        help="The experiment that we want to run",
    )                        
    parser.add_argument(
        "--directory",
        default="/content/drive/MyDrive/PhD/reproducibility/CAIRO_github/CAIRO-experiment/",
        help="The directory where the files are stored",
    )
    parser.add_argument(
        "--unify_subgroups",
        type=bool,
        default=True,
        help="Whether or not to paraphrase the prompts.",
    )   
    parser.add_argument("-split", "--split_list", nargs="+", default=[])
    parser.add_argument("-group", "--group_list", nargs="+", default=[])
    parser.add_argument("-model", "--model_list", nargs="+", default=[])
    parser.add_argument("-prompting", "--prompting_list", nargs="+", default=[])

    return parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    def compute_bias(df, group):
        bias = 0
        toxicity_group = df[df.columns[4]][df["group"] == group].mean()

        for subgroup in df["subgroup"].unique():
            toxicity_subgroup = df[df.columns[4]][(df["group"] == group) & (df["subgroup"] == subgroup)].mean()
            bias += abs(toxicity_group-toxicity_subgroup)
        return bias
    
    def data_augmentation_effect(df_all_prompts_models, df_all_seeds, paraphrasing_model, prompting, model_name, split, seed, group):
        if len(df_all_prompts_models) > 0:
            for num_prompts in range(1,7):
                for with_replacement in [True]:
                    list_prompts = ["original_prompts"]
                    remaining_prompts = list(df_all_prompts_models["title"].unique())
                    if "original_prompts" in remaining_prompts:
                        remaining_prompts.remove("original_prompts")

                    if with_replacement:
                        prompts_combinations = list(itertools.combinations_with_replacement(remaining_prompts, num_prompts-1))
                    else:
                        prompts_combinations = list(itertools.combinations(remaining_prompts, num_prompts-1))
                    
                    sample_id=0
                    for i in range(len(prompts_combinations)):
                        list_prompts=list(prompts_combinations[i])+["original_prompts"]
                        df=pd.DataFrame()
                        for prompt in list_prompts:
                            df=pd.concat([df, df_all_prompts_models[df_all_prompts_models["title"] == prompt]], ignore_index=True)
                        df_all_seeds = df_all_seeds.append({'Group': group,'Hurtfulness': df["honest_score"].mean() if prompting == "HONEST" else None, 'Bias': compute_bias(df, group), 'Data augmentation': True, 'Split': split,
                                                            'Prompting': prompting, 'Replacement': with_replacement,'Group': group, 'Model': model_name.replace("/", "_"),'Paraphrasing model': paraphrasing_model,'Sample number': sample_id, 'Unify subgroups': args.unify_subgroups,
                                                            'Seed': seed, 'Num prompts': num_prompts}, ignore_index = True)     
                        sample_id += 1
        return df_all_seeds
    
    df_all_seeds = pd.DataFrame()
    seeds=range(1,6)
    prompting_models=['original_prompts']
    if args.paraphrasing_model == 'Chatgpt':
        prompting_models += ['prompts_gpt-35-turbo-16k_1','prompts_gpt-35-turbo-16k_2','prompts_gpt-35-turbo-16k_3','prompts_gpt-35-turbo-16k_4','prompts_gpt-35-turbo-16k_5']
    if args.paraphrasing_model == 'Mistral':
        prompting_models += ['prompts_mistralai_Mistral-7B-Instruct-v0.2_1','prompts_mistralai_Mistral-7B-Instruct-v0.2_2','prompts_mistralai_Mistral-7B-Instruct-v0.2_3','prompts_mistralai_Mistral-7B-Instruct-v0.2_4','prompts_mistralai_Mistral-7B-Instruct-v0.2_5']
    if args.paraphrasing_model == 'Llama':
        prompting_models += ['prompts_meta-llama_Llama-2-7b-chat-hf_1','prompts_meta-llama_Llama-2-7b-chat-hf_2','prompts_meta-llama_Llama-2-7b-chat-hf_3','prompts_meta-llama_Llama-2-7b-chat-hf_4','prompts_meta-llama_Llama-2-7b-chat-hf_5']
    else:
        prompting_models += ['prompts_gpt-35-turbo-16k_1','prompts_gpt-35-turbo-16k_2','prompts_gpt-35-turbo-16k_3','prompts_gpt-35-turbo-16k_4','prompts_gpt-35-turbo-16k_5',
                            'prompts_mistralai_Mistral-7B-Instruct-v0.2_1','prompts_mistralai_Mistral-7B-Instruct-v0.2_2','prompts_mistralai_Mistral-7B-Instruct-v0.2_3','prompts_mistralai_Mistral-7B-Instruct-v0.2_4','prompts_mistralai_Mistral-7B-Instruct-v0.2_5',
                            'prompts_meta-llama_Llama-2-7b-chat-hf_1','prompts_meta-llama_Llama-2-7b-chat-hf_2','prompts_meta-llama_Llama-2-7b-chat-hf_3','prompts_meta-llama_Llama-2-7b-chat-hf_4','prompts_meta-llama_Llama-2-7b-chat-hf_5']

    if args.experiment == "collect_all_csv_files":
        for prompting, experiment, model_name, split, group, paraphrasing_model in itertools.product(args.prompting_list, ["robustness_effect", "data_augmentation_effect"], args.model_list, ["valid", "test"], ["race_ethnicity", "religion", "gender_and_sex","gender","race","religious_ideology"], ["Chatgpt", "Mistral", "Llama","Chatgpt_Llama_Mistral", "None"]):
        # for prompting in args.prompting_list:  
        #     for experiment in ["robustness_effect", "data_augmentation_effect"]:
        #         for model_name in args.model_list:
        #             for split in ["valid", "test"]:
        #                 for group in ["race_ethnicity", "religion", "gender_and_sex","gender","race","religious_ideology"]: 
        #                     for paraphrasing_model in ["Chatgpt", "Mistral", "Llama","Chatgpt_Llama_Mistral", "None"]:
                                csv_directory = (
                                    args.directory
                                    + "seed_1"
                                    + "/output/"
                                    + "/"                   
                                ) 
                                if prompting == "holistic" and group in ["gender_and_sex", "religion"]:

                                    file_name = (
                                        csv_directory
                                        + str(prompting)
                                        + "_"
                                        + experiment
                                        + "_"
                                        + str(model_name).replace("/", "_")
                                        + "_"
                                        + str(paraphrasing_model)
                                        + "_"
                                        + str(split)
                                        + "_"
                                        + str(group)
                                        + ".csv"
                                    )                                
                                
                                else:
                                    file_name = (
                                        csv_directory
                                        + str(prompting)
                                        + "_"
                                        + experiment
                                        + "_"
                                        + str(model_name).replace("/", "_")
                                        + "_"
                                        + str(paraphrasing_model)
                                        + "_"
                                        + str(split)
                                        + "_"
                                        + str(group)
                                        + ".csv"
                                    )
                                if os.path.exists(file_name):
                                    print(file_name)
                                    if os.stat(file_name).st_size < 10:
                                        continue
                                    df = pd.read_csv(file_name,lineterminator='\n', error_bad_lines=False)
                                    if 'prompt list' in df.columns:
                                        df=df.drop(['prompt list'], axis=1) 
                                    df_all_seeds=pd.concat([df_all_seeds, df], ignore_index=True)

            df_all_seeds.to_csv(
                "./output/"
                + str(prompting)
                + "_"
                + "everything.csv",
                index=False,
            )

    for prompting, model_name, split, group, seed in itertools.product(args.prompting_list, args.model_list, args.split_list, args.group_list, seeds):
    # for prompting in args.prompting_list:
    #     for model_name in args.model_list:
    #         for split in args.split_list:
    #             for seed in seeds:
                    for group in args.group_list:
                        df_all_prompts_models=pd.DataFrame()
                        for prompting_model in prompting_models:
                            csv_directory = (
                                args.directory
                                +"seed_"
                                + str(seed)
                                + "/output/"
                                + str(prompting)
                                + "_" + str(split)  
                                + "/"                   
                            ) 
                            file_name = (
                                csv_directory
                                + model_name.replace("/", "_")
                                + "_"
                                + str(group)
                                + "_"
                                + str(prompting_model)
                                + ".csv"
                            )  
                            print(file_name)   
                            if os.path.exists(file_name):
                                print(prompting,model_name,split,seed,group,prompting_model)
                                if os.stat(file_name).st_size < 10000:
                                    continue
                                df = pd.read_csv(file_name,lineterminator='\n', error_bad_lines=False)  
                                df = df.drop(df[(df["title"] == "prompts_gpt-35-turbo-16k_10")].index)

                                if prompting == "holistic" and args.unify_subgroups:
                                    if group == "gender_and_sex":
                                        df=df.drop(df[(df["subgroup"] != "queer")&(df["subgroup"] != "binary")&(df["subgroup"] != "sex")&(df["subgroup"] != "descriptors")].index)
                                    elif group == "religion":
                                        print("before ", len(df))
                                        df_filtered = pd.read_csv("./prompts/holistic/" + "filtered_religion_subgroups_" + split + ".csv")
                                        df=df[~df_filtered["prompt"]]
                                        print("after ", len(df))

                                df_all_prompts_models=pd.concat([df_all_prompts_models, df], ignore_index=True)
                                if args.experiment == "robustness_effect":
                                    df_all_seeds = df_all_seeds.append({'Group': group,'Hurtfulness': df["honest_score"].mean() if prompting == "HONEST" else None, 'Bias': compute_bias(df, group), 
                                                                        'Prompting': prompting, 'Prompt': prompting_model,'Group': group,'Model': model_name.replace("/", "_"), 'Unify subgroups': args.unify_subgroups,
                                                                        'Seed': seed, 'Data augmentation': False, 'Split': split}, ignore_index = True)     
                                    print("--- %s seconds ---" % (time.time() - start_time))

                        if args.experiment == "data_augmentation_effect":
                                            df_all_seeds=data_augmentation_effect(df_all_prompts_models, df_all_seeds, args.paraphrasing_model, prompting, model_name, split, seed, group)
                                            print("--- %s seconds ---" % (time.time() - start_time))

    print(df_all_seeds)
    df_all_seeds.to_csv(
        "./output/"
        + str(prompting)
        + "_"
        + str(args.experiment)
        + "_"
        + str(model_name.replace("/", "_"))
        + "_"
        + str(args.paraphrasing_model)
        + "_"
        + str(split)
        + "_"
        + str(group)
        + ".csv",
        index=False,
    ) 