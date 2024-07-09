import warnings
import pandas as pd
import os
import torch
from argparse import ArgumentParser
import time
import warnings
import math
warnings.simplefilter(action='ignore', category=FutureWarning)
import warnings
import math
from scipy.stats import pearsonr
import seaborn as sns
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
        default="Chatgpt",
        help="The model used to generate the paraphrases",
    ) 

    parser.add_argument(
        "--experiment",
        choices=[
            "collect_everything",
            "collect_all_paraphrasing_models",
        ],
        default="collect_all_paraphrasing_models",
        help="The experiment that we want to run",
    )                  
    parser.add_argument(
        "--directory",
        default="/content/drive/MyDrive/PhD/reproducibility/CAIRO_github/CAIRO-experiment/",
        help="The directory where the files are stored",
    )

    parser.add_argument("-model", "--model_list", nargs="+", default=[])
    parser.add_argument("-prompting", "--prompting_list", nargs="+", default=[])
    parser.add_argument("-split", "--group_list", nargs="+", default=[])
    parser.add_argument("-group", "--split_list", nargs="+", default=[])

    return parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    if args.experiment == "collect_all_paraphrasing_models":
            df = pd.DataFrame()
            for prompting in args.prompting_list:   
                csv_directory = (
                    args.directory
                    + "seed_1"
                    + "/output/"
                    + "/"                   
                ) 
                file_name = (
                    csv_directory
                    + str(prompting)
                    + "_"
                    + "everything.csv"
                )
                print(file_name)
                if os.path.exists(file_name):
                    df_current = pd.read_csv(file_name,lineterminator='\n', error_bad_lines=False)  
                    df=pd.concat([df, df_current], ignore_index=True)

            print(df)
            sns.set_style("darkgrid")
            df["Group"].replace({"nationality": "Nationality", "sexual_orientation": "Sexual-orientation", "gender_and_sex": "Gender", "religion": "Religion", "religious_ideology": "Religion", "race_ethnicity": "Race", "race": "Race","gender": "Gender"}, inplace=True)
            sns.set(font_scale = 1.5)

            df=df[(df["Model"]==args.model_list[0].replace("/", "_"))& (df["Replacement"]==True) & (df["Paraphrasing model"]==args.paraphrasing_model) & (df["Group"]==args.group_list[0]) & (df["Split"]==args.split_list[0])]
            print(df)
            df["BOLD bias"]=None
            df["Holistic bias"]=None
            df["HONEST bias"]=None
            df["HONEST hurtfulness"]=None


            df.loc[df["Prompting"]=="BOLD","BOLD bias"]=df[df["Prompting"]=="BOLD"]["Bias"]
            df.loc[df["Prompting"]=="HONEST","HONEST bias"]=df[df["Prompting"]=="HONEST"]["Bias"]
            df.loc[df["Prompting"]=="HONEST","HONEST hurtfulness"]=df[df["Prompting"]=="HONEST"]["Hurtfulness"]
            df.loc[df["Prompting"]=="holistic","Holistic bias"]=df[df["Prompting"]=="holistic"]["Bias"]

            df["Replacement"].replace({'1.0': True}, inplace=True)
            df["Replacement"].replace({'True': True}, inplace=True)
            df["Replacement"].replace({'False': False}, inplace=True)

            df["Data augmentation"].replace({'1.0': True}, inplace=True)

            df_new = pd.DataFrame()
            for num, model, replacement, split, paraphrasing_model in itertools.product(df["Num prompts"].unique(), df["Model"].unique(), df["Replacement"].unique(), df["Split"].unique(), df["Paraphrasing model"].unique()):
                my_df=df[(df["Num prompts"]==num)&(df["Replacement"]==replacement)&(df["Paraphrasing model"]==paraphrasing_model) &(df["Model"]==model) & (df["Split"]==split)]
                for sample_num in my_df["Sample number"].unique():
                    my_df=df[(df["Sample number"]==sample_num)]
                    if math.isnan(num):
                        continue
                    print({'Num prompts': num,'Model': model,'Group':'Gender','Paraphrasing model':paraphrasing_model,
                                            'Replacement': replacement,'Sample number': sample_num,
                                            'Holistic bias': my_df[my_df["Group"]=="Gender"]["Holistic bias"].mean(),
                                            'HONEST hurtfulness': my_df[my_df["Group"]=="Gender"]["HONEST hurtfulness"].mean(),
                                            'HONEST bias': my_df[my_df["Group"]=="Gender"]["HONEST bias"].mean(),'Split': split,
                                            'BOLD bias': my_df[my_df["Group"]=="Gender"]["BOLD bias"].mean()})
                    df_new = df_new.append({'Num prompts': num,'Model': model,'Group':'Gender','Paraphrasing model':paraphrasing_model,
                                            'Replacement': replacement,'Sample number': sample_num,
                                            'Holistic bias': my_df[my_df["Group"]=="Gender"]["Holistic bias"].mean(),
                                            'HONEST hurtfulness': my_df[my_df["Group"]=="Gender"]["HONEST hurtfulness"].mean(),
                                            'HONEST bias': my_df[my_df["Group"]=="Gender"]["HONEST bias"].mean(),'Split': split,
                                            'BOLD bias': my_df[my_df["Group"]=="Gender"]["BOLD bias"].mean()}, ignore_index = True)

                    df_new = df_new.append({'Num prompts': num,'Model': model,'Group':'Religion','Paraphrasing model':paraphrasing_model,
                                            'Replacement': replacement,'Sample number': sample_num,
                                            'Holistic bias': my_df[my_df["Group"]=="Religion"]["Holistic bias"].mean(),
                                            'HONEST hurtfulness': 0,
                                            'HONEST bias': 0 ,'Split': split,
                                            'BOLD bias': my_df[my_df["Group"]=="Religion"]["BOLD bias"].mean()}, ignore_index = True)

                    df_new = df_new.append({'Num prompts': num,'Model': model,'Group':'Race','Paraphrasing model':paraphrasing_model,
                                            'Replacement': replacement,'Sample number': sample_num,
                                            'Holistic bias': my_df[my_df["Group"]=="Race"]["Holistic bias"].mean(),
                                            'HONEST hurtfulness': 0,
                                            'HONEST bias': 0,'Split': split,
                                            'BOLD bias': my_df[my_df["Group"]=="Race"]["BOLD bias"].mean()}, ignore_index = True)



            df_new=df_new.dropna()
            print(df_new)
            df_new.to_csv(
                "./output/"
                + str(args.experiment)
                + "_"
                + str(args.model_list[0].replace("/", "_"))
                + "_"
                + str(args.paraphrasing_model)
                + "_"
                + str(args.group_list[0])
                + "_"
                + str(args.split_list[0])
                + ".csv",
                index=False,
            ) 


    if args.experiment == "collect_everything":

        df_new = pd.DataFrame()
        for split in ["valid", "test"]:
            for group in ["Gender", "Religion", "Race"]:
                for model in args.model_list:
                    for paraphrasing_model in ["Chatgpt", "Mistral", "Llama","Chatgpt_Llama_Mistral"]:  
                        print(model, paraphrasing_model) 
                        csv_directory = (
                            args.directory
                            + "seed_1"
                            + "/output/"
                            + "/"                   
                        ) 
                        file_name = (
                            csv_directory
                            + "collect_all_paraphrasing_models"
                            + "_"
                            + str(model.replace("/", "_"))
                            + "_"
                            + str(paraphrasing_model)
                            + "_"
                            + str(group)
                            + "_"
                            + str(split)
                            + ".csv"
                        )
                        print(file_name)
                        if os.path.exists(file_name):
                            df_current = pd.read_csv(file_name,lineterminator='\n', error_bad_lines=False)  
                            df_new=pd.concat([df_new, df_current], ignore_index=True)


        df_new.to_csv(
            "./output/"
            + "all_models"
            + ".csv",
            index=False,
        ) 


