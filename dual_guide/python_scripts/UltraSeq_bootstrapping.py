import numpy as np
import pandas as pd
import math
from UltraSeq_metrics_functions import *
from scipy.stats import rankdata

''' This module contains steps for bootstrapping. It has function for bootstrapping KTHC mice and then within mouse gRNAs and control (KT) mice and with within mouse gRNAs.
    It has functions to generate gRNA and gene-level summaries of the tumor metrics.
'''

def Nested_Boostrap_Index_single(input_dic,input_sample_list,input_factor):
    # for sampling KTHC
    # I first sample mouse
    temp_list = np.random.choice(input_sample_list,len(input_sample_list),replace = True)
    temp_coho = []
    for y in temp_list: # within each mouse
        temp_array = input_dic.get(y) # array of tuple, each is a (gRNA, clonal_barcode)
        temp_resampled = np.random.choice(temp_array,round(len(temp_array)/input_factor),replace = True)
        temp_coho = np.concatenate([temp_coho,temp_resampled])
    return(temp_coho)  

def Nested_Boostrap_Index_Special_single(input_dic,input_df,input_total_gRNA_number,input_sample_list,input_factor):
    # for sampling KT
    # I first sample mouse
    temp_coho = []
    while len(set(input_df.loc[temp_coho].gRNA)) < input_total_gRNA_number: # resample until all the gRNAs have been resampled
        temp_list = np.random.choice(input_sample_list,len(input_sample_list),replace = True)
        temp_coho = []
        for y in temp_list: # within each mouse
            temp_array = input_dic.get(y) # all the tumor (gRNAs) indices in the mouse
            temp_resampled = np.random.choice(temp_array,round(len(temp_array)/input_factor),replace = True)
            temp_coho = np.concatenate([temp_coho,temp_resampled]) 
    return(temp_coho)

def Generate_Final_Summary_Dataframe(input_df,trait_of_interest):
    # gRNA level summary
    #temp_summary = input_df.groupby(['gRNA','Targeted_gene_name','Numbered_gene_name'],as_index = False).apply(Cal_Bootstrapping_Summary,(trait_of_interest))
    temp_summary = input_df.groupby(['gRNA','Targeted_gene_name'],as_index = False).apply(Cal_Bootstrapping_Summary,(trait_of_interest))
    temp_output_df = temp_summary
    for temp_trait in trait_of_interest:
        temp_name0 = temp_trait + '_fraction_greater_than_one'
        temp_name1 = temp_trait + '_pvalue'
        temp_name2 = temp_name1 + '_FDR'
        temp_name3 = temp_name1 + '_twoside'
        temp_name4 = temp_name1 + '_twoside_FDR'
        temp_output_df[temp_name1] = temp_output_df.apply(lambda x: min(x[temp_name0],1-x[temp_name0]), axis=1) # 
        temp_output_df[temp_name2] = fdr(temp_output_df[temp_name1])
        temp_output_df[temp_name3] = temp_output_df[temp_name1]*2
        temp_output_df[temp_name4] = fdr(temp_output_df[temp_name3])
    return(temp_output_df)

def Generate_Gene_Level_Summary_Dataframe(input_df, trait_of_interest):
    #temp_summary = input_df.groupby(['Targeted_gene_name','Level2_bootstrap_id'],as_index = False).apply(Cal_Combined_Gene_Effect_v2,(trait_of_interest))
    temp_summary = input_df.groupby(['Targeted_gene_name','Level3_bootstrap_id'],as_index = False).apply(Cal_Combined_Gene_Effect_v2,(trait_of_interest))
    temp_output_df = temp_summary.groupby(['Targeted_gene_name'], as_index=False).apply(Cal_Bootstrapping_Summary_V2,(trait_of_interest))
    for temp_trait in trait_of_interest:
        temp_name0 = temp_trait + '_fraction_greater_than_one'
        temp_name1 = temp_trait + '_pvalue'
        temp_name2 = temp_name1 + '_FDR'
        temp_name3 = temp_name1 + '_twoside'
        temp_name4 = temp_name1 + '_twoside_FDR'
        temp_output_df[temp_name1] = temp_output_df.apply(lambda x: min(x[temp_name0],1-x[temp_name0]), axis=1) # 
        temp_output_df[temp_name2] = fdr(temp_output_df[temp_name1])
        temp_output_df[temp_name3] = temp_output_df[temp_name1]*2
        temp_output_df[temp_name4] = fdr(temp_output_df[temp_name3])
    return(temp_output_df)