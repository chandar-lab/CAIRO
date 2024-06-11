import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
import torch
import json
import numpy as np
from argparse import ArgumentParser
import random
from random import choices
import time
import random, seaborn as sns, matplotlib.pyplot as plt
import warnings
import math
warnings.simplefilter(action='ignore', category=FutureWarning)
import warnings
import math
warnings.simplefilter(action='ignore', category=FutureWarning)

from scipy.stats import pearsonr
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
            "compute_correlation",
            "collect_all_paraphrasing_models",
        ],
        default="collect_all_paraphrasing_models",
        help="The experiment that we want to run",
    )                  
    parser.add_argument(
        "--account",
        choices=[
            "abdel1", "olewicki"
        ],
        default="abdel1",
        help="The Compute Canada account that we work on",
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
                    "/scratch/" + args.account + "/bias_metrics/holistic_bias/seed_1"
                    + "/output/"
                    + "/"                   
                ) 
                file_name = (
                    csv_directory
                    + str(prompting)
                    + "_"
                    + "everything_reproducibility_2.csv"
                )
                print(file_name)
                if os.path.exists(file_name):
                    print("Adeene geet")
                    df_current = pd.read_csv(file_name,lineterminator='\n', error_bad_lines=False)  
                    df=pd.concat([df, df_current], ignore_index=True)

            print(df)
            sns.set_style("darkgrid")
            df["Group"].replace({"nationality": "Nationality", "sexual_orientation": "Sexual-orientation", "gender_and_sex": "Gender", "religion": "Religion", "religious_ideology": "Religion", "race_ethnicity": "Race", "race": "Race","gender": "Gender"}, inplace=True)
            sns.set(font_scale = 1.5)

            # df=df[(df["Model"]==args.model_list[0].replace("/", "_"))& (df["Replacement"]==True) & (df["Paraphrasing model"]==args.paraphrasing_model) & (df["Group"]==args.group_list[0]) & (df["Split"]==args.split_list[0])].drop(['prompt list'], axis=1)
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
            for num in df["Num prompts"].unique():
                for model in df["Model"].unique():
                    for replacement in df["Replacement"].unique():
                        for split in df["Split"].unique():
                            for paraphrasing_model in df["Paraphrasing model"].unique():
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
                + "_reproducibility_2.csv",
                index=False,
            ) 


    if args.experiment == "compute_correlation":

        df_new = pd.DataFrame()
        for split in ["valid", "test"]:
            for group in ["Gender", "Religion", "Race"]:
                for model in args.model_list:
                    for paraphrasing_model in ["Chatgpt", "Mistral", "Llama","Chatgpt_Llama_Mistral"]:  
                        print(model, paraphrasing_model) 
                        csv_directory = (
                            "/scratch/" + args.account + "/bias_metrics/holistic_bias/seed_1"
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
                            + "_reproducibility_2.csv"
                        )
                        print(file_name)
                        if os.path.exists(file_name):
                            print("Adeene geet")
                            df_current = pd.read_csv(file_name,lineterminator='\n', error_bad_lines=False)  
                            df_new=pd.concat([df_new, df_current], ignore_index=True)


        df_new.to_csv(
            "./output/"
            + "all_models_reproducibility_2"
            + ".csv",
            index=False,
        ) 

        df_new_3 = pd.DataFrame()
        for num in df_new["Num prompts"].unique():
            for group in df_new["Group"].unique():
                for paraphrasing_model in df_new["Paraphrasing model"].unique():
                    for replacement in df_new["Replacement"].unique():
                        corr_holistic_bold_max, corr_honest_holistic_max, corr_honest_bold_max=[-99,0],[-99,0],[-99,0]
                        corr_holistic_bold_min, corr_honest_holistic_min, corr_honest_bold_min=[99,0],[99,0],[99,0]
                        corr_holistic_bold_sum, corr_honest_holistic_sum, corr_honest_bold_sum=0,0,0
                        for sample_num in df_new["Sample number"].unique():
                            
                            my_df=df_new[(df_new["Num prompts"]==num)&(df_new["Group"]==group)&(df_new["Paraphrasing model"]==paraphrasing_model)&(df_new["Replacement"]==replacement)&(df_new["Sample number"]==sample_num)]
                            if (len(my_df['Holistic bias'])<2) or (len(my_df['BOLD bias'])<2) or (len(my_df['HONEST hurtfulness'])<2) or (len(my_df['HONEST bias'])<2):
                                continue
                            # print(my_df, len(my_df['Holistic bias']), len(my_df['BOLD bias']), len(my_df['HONEST hurtfulness']), len(my_df['HONEST bias']))
                            corr_holistic_bold=pearsonr(my_df['Holistic bias'], my_df['BOLD bias'])
                            corr_honest_holistic=pearsonr(my_df['HONEST hurtfulness'], my_df['Holistic bias'])
                            corr_honest_bold=pearsonr(my_df['HONEST hurtfulness'], my_df['BOLD bias'])

                            corr_holistic_bold_sum+=corr_holistic_bold[0]
                            corr_honest_holistic_sum+=corr_honest_holistic[0]
                            corr_honest_bold_sum+=corr_honest_bold[0]

                            # print(num, replacement, sample_num,corr_honest_bold[0],corr_holistic_bold_min[0])

                            if corr_holistic_bold[0]>corr_holistic_bold_max[0]:
                                corr_holistic_bold_max=corr_holistic_bold

                            if corr_holistic_bold[0]<corr_holistic_bold_min[0]:
                                corr_holistic_bold_min=corr_holistic_bold

                    ###########################

                            if corr_honest_holistic[0]>corr_honest_holistic_max[0]:
                                corr_honest_holistic_max=corr_honest_holistic

                            if corr_honest_holistic[0]<corr_honest_holistic_min[0]:
                                corr_honest_holistic_min=corr_honest_holistic

                    # #########################

                            if corr_honest_bold[0]>corr_honest_bold_max[0]:
                                corr_honest_bold_max=corr_honest_bold

                            if corr_honest_bold[0]<corr_honest_bold_min[0]:
                                corr_honest_bold_min=corr_honest_bold

                    #########################
                        # df_new_3 = df_new_3.append({'Num prompts': num,'Replacement': replacement,'Type':'Average','Correlation': (corr_holistic_bold[0]+corr_honest_holistic[0]+corr_honest_bold[0])/3}, ignore_index = True)
                        # print(corr_honest_bold_min)
                        df_new_3 = df_new_3.append({'Group':group,'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Oracle','Type':'Holistic-BOLD','Correlation': corr_holistic_bold_max[0],'P-value': corr_holistic_bold_max[1]}, ignore_index = True)
                        df_new_3 = df_new_3.append({'Group':group,'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Oracle','Type':'HONEST-Holistic','Correlation': corr_honest_holistic_max[0],'P-value': corr_honest_holistic_max[1]}, ignore_index = True)
                        df_new_3 = df_new_3.append({'Group':group,'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Oracle','Type':'HONEST-BOLD','Correlation': corr_honest_bold_max[0],'P-value': corr_honest_bold_max[1]}, ignore_index = True)
                        df_new_3 = df_new_3.append({'Group':group,'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Oracle','Type':'Average correlation','Correlation': 1/3*(corr_holistic_bold_max[0]+corr_honest_holistic_max[0]+corr_honest_bold_max[0]),'P-value': 1/3*(corr_holistic_bold_max[1]+corr_honest_holistic_max[1]+corr_honest_bold_max[1])}, ignore_index = True)


                        # df_new_3 = df_new_3.append({'Group':group,'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Lowest corr','Type':'Holistic-BOLD','Correlation':  corr_holistic_bold_min[0],'P-value': corr_holistic_bold_min[1]}, ignore_index = True)
                        # df_new_3 = df_new_3.append({'Group':group,'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Lowest corr','Type':'HONEST-Holistic','Correlation':  corr_honest_holistic_min[0],'P-value': corr_honest_holistic_min[1]}, ignore_index = True)
                        # df_new_3 = df_new_3.append({'Group':group, 'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Lowest corr','Type':'HONEST-BOLD','Correlation':  corr_honest_bold_min[0],'P-value': corr_honest_bold_min[1]}, ignore_index = True)

                        # df_new_3 = df_new_3.append({'Group':group,'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Random','Type':'Holistic-BOLD','Correlation':  corr_holistic_bold[0],'P-value': corr_holistic_bold[1]}, ignore_index = True)
                        # df_new_3 = df_new_3.append({'Group':group,'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Random','Type':'HONEST-Holistic','Correlation':  corr_honest_holistic[0],'P-value': corr_honest_holistic[1]}, ignore_index = True)
                        # df_new_3 = df_new_3.append({'Group':group, 'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Random','Type':'HONEST-BOLD','Correlation':  corr_honest_bold[0],'P-value': corr_honest_bold[1]}, ignore_index = True)

                        df_new_3 = df_new_3.append({'Group':group,'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Average','Type':'Holistic-BOLD','Correlation':  corr_holistic_bold_sum/len(df_new["Sample number"].unique()),'P-value': None}, ignore_index = True)
                        df_new_3 = df_new_3.append({'Group':group,'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Average','Type':'HONEST-Holistic','Correlation':  corr_honest_holistic_sum/len(df_new["Sample number"].unique()),'P-value': None}, ignore_index = True)
                        df_new_3 = df_new_3.append({'Group':group, 'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Average','Type':'HONEST-BOLD','Correlation':  corr_honest_bold_sum/len(df_new["Sample number"].unique()),'P-value':None}, ignore_index = True)
                        df_new_3 = df_new_3.append({'Group':group, 'Num prompts': num,'Paraphrasing model': paraphrasing_model,'Replacement': replacement,'Method':'Average','Type':'Average correlation','Correlation':  1/3*((corr_holistic_bold_sum/len(df_new["Sample number"].unique())) + (corr_honest_holistic_sum/len(df_new["Sample number"].unique())) + (corr_honest_bold_sum/len(df_new["Sample number"].unique()))),'P-value':None}, ignore_index = True)


        print("--- %s seconds ---" % (time.time() - start_time))
        print("Done")

        df_new_3.to_csv(
            "./output/"
            + str(args.experiment)
            + "_reproducibility_2.csv",
            index=False,
        ) 