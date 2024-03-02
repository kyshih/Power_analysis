#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from UltraSeq_MetricsFunction import *
from Ultra_Seq_Boostrapping import *

## Functions
def Generate_treatment_effect_df(input_ref_df, input_treatment_df,trait_of_interest):
    ref_trait = ['Type','Targeted_gene_name','Numbered_gene_name','Bootstrap_id','gRNA']
    temp_combined = pd.merge(input_ref_df[trait_of_interest+ref_trait],
                             input_treatment_df[trait_of_interest+ref_trait],
                            on=ref_trait, how ='outer', suffixes=('_ref', '_treatment'))
    temp_combined = temp_combined.fillna(0)
    for x in trait_of_interest:
        temp_combined[x] = temp_combined.apply(lambda y: y[x+'_treatment']/y[x + '_ref'],axis=1)
    temp_final_df = temp_combined[ref_trait+trait_of_interest]
    return(temp_final_df)

# -----
def main():
    parser = argparse.ArgumentParser(description='A function to do calculate treatment effect using bootstrapping data')
    parser.add_argument("--a0", required=True, help="Address of reference data")
    parser.add_argument("--a1", required=True, help="Address of treatment data")
    parser.add_argument("--o1", required=True, help="This the output address for summary data")
    parser.add_argument('--l1', nargs='+', required=False, help="A list of quantile that I want to calculate tumor size quantile: 50 60 70 80 90 95 97 99")
    
    # data input
    args = parser.parse_args()
    
    ref_address  = args.a0
    treatment_address  = args.a1
    output_address = args.o1
    temp_q = [int(x) for x in args.l1]

    ref_df = pd.read_csv(ref_address) # ref data frame
    treatment_df = pd.read_csv(treatment_address) # treatment data frame

    # trait list 
    temp_trait_list = ['LN_mean_relative','Geo_mean_relative','TTB_normalized_relative','TTN_normalized_relative','95_percentile_relative'] + [str(x) + '_percentile_relative' for x in temp_q]
    temp_trait_list = list(set(temp_trait_list))
    
    # Generate relative score df 
    temp_ratio = Generate_treatment_effect_df(ref_df, treatment_df,temp_trait_list)
    # Generate summary statistics
    Final_summary_df = Generate_Final_Summary_Dataframe(temp_ratio,temp_trait_list)
    Final_gene_summary_df = Generate_Gene_Level_Summary_Dataframe(temp_ratio,temp_trait_list)
    # rename columns
    temp_rename_dic = {}
    for x in Final_summary_df.columns:
        if 'relative' in x:
            temp_rename_dic[x] = x.replace('_relative', '_relative_score')
    Final_summary_df.rename(columns = temp_rename_dic, inplace=True)

    temp_rename_dic = {}
    for x in Final_gene_summary_df.columns:
        if 'relative' in x:
            temp_rename_dic[x] = x.replace('_relative', '_relative_score')
    Final_gene_summary_df.rename(columns = temp_rename_dic, inplace=True)
    # output data
    Final_summary_df.to_csv(output_address+'_sgRNA_level.csv',index = False)
    Final_gene_summary_df.to_csv(output_address+'_gene_level.csv',index = False)
    print(f"All steps finished") 

if __name__ == "__main__":
    main() 
