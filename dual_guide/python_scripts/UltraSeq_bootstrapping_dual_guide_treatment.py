#!/usr/bin/env python
# coding: utf-8

# # Deep sequencing analysis
# # Use bootstrapping to study generate the summary statistics
# _________

# ## 1 Functions and module

# ### 1.1 Modules

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import numpy as np
import copy
import scipy
import random
import argparse
from scipy.stats import rankdata

### Functions
def Bootstrapping_Final_df(raw_df,input_sample_list1,input_sample_list2,cell_number_cutoff,percentile_list,input_control_gRNA_list,number_of_replicate,input_total_gRNA_number):
    # return tumor size metric of each bootstrap cycle
    # experimental mouse
    temp_ref_df1 = Generate_ref_input_df(raw_df,input_sample_list1,cell_number_cutoff)
    # Control mouse
    temp_ref_df2 = Generate_ref_input_df(raw_df,input_sample_list2,cell_number_cutoff)
    temp_final_df_observed = Calculate_Relative_Normalized_Metrics(temp_ref_df1,temp_ref_df2,percentile_list,input_control_gRNA_list)
    temp_final_df_observed['Bootstrap_id'] = ['Real']*temp_final_df_observed.shape[0] # a new columns named Bootstrap_id. The values of the column are 'Real'
    temp_out_df = temp_final_df_observed
    if number_of_replicate!=0:
        # experimental
        Mouse_index_dic_1 = Generate_Index_Dictionary(temp_ref_df1) # has tumor size info for each SampleID
        # control
        Mouse_index_dic_2 = Generate_Index_Dictionary(temp_ref_df2)
        for bootstrap_cycle in range(number_of_replicate):
            # resampling the experimental mice
            x = Nested_Boostrap_Index_single(Mouse_index_dic_1)
            temp_bootstrap_df_1 = temp_ref_df1.loc[x]
            # resampleing the control mice
            y = Nested_Boostrap_Index_Special_single(Mouse_index_dic_2,temp_ref_df2,input_total_gRNA_number)
            temp_bootstrap_df_2 = temp_ref_df2.loc[y]
            # Experimental mice normalized to control mice and normalize sgTS to sgInert
            temp_metric_df = Calculate_Relative_Normalized_Metrics(temp_bootstrap_df_1,temp_bootstrap_df_2,percentile_list,input_control_gRNA_list)
            temp_metric_df['Bootstrap_id'] = ['B'+str(bootstrap_cycle)]*temp_metric_df.shape[0]
            temp_out_df = pd.concat([temp_out_df.reset_index(drop=True),temp_metric_df.reset_index(drop=True)]) # dropping the old index column. concatenate df along the rows
    return(temp_out_df)

def Nested_Boostrap_Index_single(input_dic):
    # input_dic has {SampleID : [row_number that corresponds to a gRNA and read counts, etc]}
    temp_sample_list = list(input_dic.keys()) # list of SampleIDs
    # I first sample mouse
    temp_list = np.random.choice(temp_sample_list,len(temp_sample_list),replace = True) # sample the SampleID with replacement
    #temp_list = np.random.choice(temp_sample_list, 20, replace = True) # resample different no. of mice
    temp_coho = []
    for y in temp_list: # within each mouse
        temp_array = input_dic.get(y) # get index of gRNA read associated of that mouse. array of tuple, each is a (gRNA, clonal_barcode)
        temp_resampled = np.random.choice(temp_array,len(temp_array),replace = True) # resample gRNA
        temp_coho = np.concatenate([temp_coho,temp_resampled])
    return(temp_coho)  

def Nested_Boostrap_Index_Special_single(input_dic,input_df,input_total_gRNA_number): # for the control mice
    temp_sample_list = list(input_dic.keys())
    # I first sample mouse
    temp_coho = []
    while len(set(input_df.loc[temp_coho].gRNA)) < input_total_gRNA_number: # stop until we reamples all the gRNA in the control mice. usually KT
        temp_list = np.random.choice(temp_sample_list,len(temp_sample_list),replace = True)
        temp_coho = []
        for y in temp_list: # within each mouse
            temp_array = input_dic.get(y) # array of tuple, each is a (gRNA, clonal_barcode)
            temp_resampled = np.random.choice(temp_array,len(temp_array),replace = True)
            temp_coho = np.concatenate([temp_coho,temp_resampled]) 
    return(temp_coho)  

def Generate_Index_Dictionary(input_df):
    # This function generate a dictionary to speed up the boostrap process
    temp_dic = {}
    temp_group = input_df.groupby(['Sample_ID'])
    # iterate through each group
    for key in temp_group.groups.keys(): # Get the keys (unique 'Sample_ID')
        # For each 'Sample_ID', get the array of indices and assign it to the dictionary
        temp_dic[key] = temp_group.get_group(key).index.values
    return(temp_dic)

def Generate_ref_input_df(input_df,input_sample_list,input_cell_cutoff):
    #return(input_df[(input_df['Cell_number']>input_cell_cutoff)&(input_df['Sample_ID'].isin(input_sample_list))])
    return(input_df[(input_df['Cell_number_0.6']>input_cell_cutoff)&(input_df['Sample_ID'].isin(input_sample_list))])

def Calculate_Relative_Normalized_Metrics(input_df1,input_df2,percentile_list,input_control_gRNA_list):
    # Cas9 mouse
    temp_df = input_df1.groupby(['gRNA','Treatment'],as_index = False).apply(Cal_Tumor_Size_simple,(percentile_list))

    # Control mouse
    temp_df2 = input_df2.groupby(['gRNA', 'Treatment'],as_index = False).apply(Cal_Tumor_Size_simple,(percentile_list))

    # normalize TTN and TTB
    # foreach gRNA, noramlize KTHC to KT
    # temp_out = Generate_Normalized_Metrics(temp_df,temp_df2,['TTN','TTB'])

    # normalize TTN and TTB
    # Apply the function to unique gRNA and Treatment combinations
    result_dfs = []
    # group by gRNA and Treatment, then for each gRNA Treatment combo, normalize KTHC to KT
    for (gRNA, Treatment), group_df in temp_df.groupby(['gRNA', 'Treatment']):
        result_df = Generate_Normalized_Metrics(group_df,
                                            temp_df2,
                                            ['TTN', 'TTB'])
        result_df['Treatment'] = Treatment  # Add a 'Treatment' column hyg or control
        result_dfs.append(result_df)
    # Concatenate the results into a single DataFrame
    # gRNA, TTN_normalized, TTB_normalized, Treatment
    temp_out = pd.concat(result_dfs, ignore_index=True)

    # merging normalized TTN and TTB with other metrics, e.g. LN mean, percentile, etc
    temp_df = temp_df.merge(temp_out,on =['gRNA', 'Treatment']) # merge
     
    # calculate relative expression to sgInert
    # e.g. TTN_normalized_relative sgTS to TTN_normalized_relative sgInert = TTN_normalized_relative
    # ignore Treatment
    # Add_Corhort_Specific_Relative_Metrics(temp_df,input_control_gRNA_list)
    # with Treatment
    # first separate the temp_df into two df based on Treatment hyg and control
    temp_df_control = temp_df.loc[temp_df.Treatment == 'control'].copy() # shallow copy
    temp_df_hyg = temp_df.loc[temp_df.Treatment == 'hyg'].copy()
    common_gRNA = temp_df.groupby('gRNA').filter(lambda x: x['Treatment'].nunique() > 1)['gRNA'].unique() # find gRNA present in both Treaments
    # only gRNAs in both Treatments
    temp_df_control = temp_df_control[temp_df_control['gRNA'].isin(common_gRNA)].copy()
    temp_df_hyg = temp_df_hyg[temp_df_hyg['gRNA'].isin(common_gRNA)].copy()
    temp_df = temp_df[temp_df['gRNA'].isin(common_gRNA)]
    # store the original index of hyg and control Treatments
    control_index = temp_df[temp_df.Treatment == 'control'].index.values
    hyg_index = temp_df[temp_df.Treatment == 'hyg'].index.values
    # make sure the gRNA order matches in both Treatments
    temp_df_hyg = temp_df_hyg.sort_values(by='gRNA').reset_index(drop=True)
    temp_df_control = temp_df_control.sort_values(by='gRNA').reset_index(drop=True)

    # normalize sgTS to sgInert under control treatment aka. sgInert-sgTS / sgInert-sgInert
    # normalize sgTS to sgInert under hyg treatment aka. sgHygR-sgTS / sgHygR-sgInert
    Add_Corhort_Specific_Relative_Metrics(temp_df_control, input_control_gRNA_list) 
    Add_Corhort_Specific_Relative_Metrics(temp_df_hyg, input_control_gRNA_list)

    # add relative and relative_scaled metrics columns to temp_df
    relative_metrics = [x for x in temp_df_control.columns.values if 'relative' in x] # all metrics relative to Inert
    normalized_to_control_metrics = [f'{x}_scaled' for x in relative_metrics] # scaled the relative metrics to control Treatment
    # add new columns of relative metrics
    temp_df[relative_metrics] = 1 # make new columns of metrics relative to Inert
    temp_df[normalized_to_control_metrics] = 1 

    # scaled all relative metrics to control Treatment
    add_corhort_specific_relative_metrics_scaled_to_untreated(temp_df_hyg, temp_df_control, relative_metrics)

    temp_df.loc[hyg_index,relative_metrics + normalized_to_control_metrics] = temp_df_hyg.loc[:, relative_metrics + normalized_to_control_metrics].values
    temp_df.loc[control_index,relative_metrics + normalized_to_control_metrics] = temp_df_control.loc[:, relative_metrics + normalized_to_control_metrics].values
    
    # annotate sample type inert or experiment
    temp_df['Type'] = temp_df.apply(lambda x: 'Inert' if (x['gRNA'] in input_control_gRNA_list) else 'Experiment',axis=1)
    # temp_df = temp_df.merge(input_df1[['gRNA','Targeted_gene_name',
    #    'Identity', 'Numbered_gene_name']].drop_duplicates(),how = 'inner',on = 'gRNA')
    temp_df = temp_df.merge(input_df1[['gRNA','Targeted_gene_name',
    'Identity', 'leftGuide', 'rightGuide', 'leftRightGuide']].drop_duplicates(),how = 'inner',on = 'gRNA')
    return(temp_df)

def add_corhort_specific_relative_metrics_scaled_to_untreated(input_df1, input_df2, metrics):
    # scale sgHygR-sgTS/median(sgHygR-sgInert) (hyg Treatment) to sgTS-sgInert/median(sgInert-sgInert) (control Treatment)
    # scaled all relative metrics in hyg Treatment to control Treatment
    # input_df1 is the experimental condition hyg
    # input_df2 is the control condition control
    for metric in metrics:
        input_df1[f'{metric}_scaled'] = input_df1[metric] / input_df2[metric]
        input_df2[f'{metric}_scaled'] = input_df2[metric] / input_df2[metric]

def Add_Corhort_Specific_Relative_Metrics(input_df,input_control_list):
    # Add relative metrics for LN_mean, GEO_mean etc using the median of inert under control treatment
    # sgTS LN_mean / sgInert LN_mean
    # for sgInert tumors, sgInert / sgInert median
    temp_sub = input_df[(input_df['gRNA'].isin(input_control_list)) ] # inert
    for temp_cname in input_df.drop(columns=['gRNA', 'Treatment'],inplace = False).columns:
        temp_name = temp_cname+'_relative'
        input_df[temp_name] = input_df[temp_cname]/temp_sub[temp_cname].median() # relative to median inert

def Cal_Bootstrapping_Summary(x,trait_of_interest):
    d = {}
    for temp_trait in trait_of_interest:
        temp0 = temp_trait + '_95P'
        temp1 = temp_trait + '_5P'
        temp2 = temp_trait +'_fraction_greater_than_one' # t_test pvalue column name
        temp3 = temp_trait +'_bootstrap_median'
        temp4 = temp_trait +'_bootstrap_mean'
        temp5 = temp_trait + '_97.5P'
        temp6 = temp_trait + '_2.5P'
        d[temp0] = x[temp_trait].quantile(0.95)
        d[temp1] = x[temp_trait].quantile(0.05)
        d[temp2] = sum(x[temp_trait]>1)/len(x[temp_trait])
        d[temp3] = x[temp_trait].mean()
        d[temp4] = x[temp_trait].median()
        d[temp5] = x[temp_trait].quantile(0.975)
        d[temp6] = x[temp_trait].quantile(0.025)
    return pd.Series(d, index=list(d.keys())) 

def Generate_Final_Summary_Dataframe(input_df,trait_of_interest): # gRNA level
    temp_summary = input_df[input_df['Bootstrap_id']!='Real'].groupby(['gRNA', 'Treatment'],as_index = False).apply(Cal_Bootstrapping_Summary,(trait_of_interest))
    temp_output_df = copy.deepcopy(input_df[input_df['Bootstrap_id'] =='Real'])
    temp_output_df = temp_output_df.merge(temp_summary, on = ['gRNA', 'Treatment'])
    for temp_trait in trait_of_interest:
        temp_name0 = temp_trait + '_fraction_greater_than_one' # t_test pvalue column name
        temp_name1 = temp_trait + '_pvalue'
        temp_name2 = temp_name1 + '_FDR'
        temp_name3 = temp_name1 + '_twoside'
        temp_name4 = temp_name1 + '_twoside_FDR'
        temp_output_df[temp_name1] = temp_output_df.apply(lambda x: min(x[temp_name0],1-x[temp_name0]), axis=1) # I dont know what this line is doing
        temp_output_df[temp_name2] = fdr(temp_output_df[temp_name1])
        temp_output_df[temp_name3] = temp_output_df[temp_name1]*2
        temp_output_df[temp_name4] = fdr(temp_output_df[temp_name3])
    return(temp_output_df)

def Generate_Gene_Level_Summary_Dataframe(input_df, trait_of_interest):
    # calculate gene effect from bootstrapping to estiamte CI, pval, and FDR
    temp_df = input_df[input_df['Bootstrap_id']!='Real'].groupby([
        'Targeted_gene_name','Bootstrap_id', 'Treatment'],as_index = False).apply(Cal_Combined_Gene_Effect_v2,(trait_of_interest))
    temp_summary = temp_df.groupby(['Targeted_gene_name', 'Treatment'],as_index = False).apply(Cal_Bootstrapping_Summary,(trait_of_interest))
    # calculate gene effect for the observed data. sample estiamte for the population parameters
    temp_output_df = input_df[input_df['Bootstrap_id'] =='Real'].groupby([
        'Targeted_gene_name', 'Treatment'],as_index = False).apply(Cal_Combined_Gene_Effect_v2,(trait_of_interest))
    temp_output_df = temp_output_df.merge(temp_summary, on = ['Targeted_gene_name', 'Treatment'])
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

def Find_Controls(input_gRNA_df, input_pattern):
# this function will find the gRNA associated with control based on the key word
# input_pattern is a regex expression 
    # return(input_gRNA_df.loc[
    #     input_gRNA_df['Targeted_gene_name'].str.contains(input_pattern, na=False, regex=True),'gRNA'].unique())
    return input_gRNA_df.loc[(input_gRNA_df['leftGuide'].str.contains(input_pattern, na=False, regex=True)) &
        (input_gRNA_df['rightGuide'].str.contains(input_pattern, na=False, regex=True)), 'gRNA'].unique()

def Cal_Tumor_Size_simple(x,input_percentile):
    # modify the metric used i.e. Cell_number, Cell_number_0.6, etc
    # return 1D labeled array for tumor size. similar to a column
    d = {}
    #temp_vect = x['Cell_number']
    temp_vect = x['Cell_number_0.6'] # fake hyg-treated inert tumors 
    #temp_vect = x['Cell_number_int_0.05'] # fake hyg-treated inert tumors 
    #temp_vect = x['Cell_number_int_0.1']
    #temp_vect = x['Cell_number_int_0.5']
    # temp_vect = x['Cell_number_int_1.1']
    #temp_vect = x['Cell_number_int_1.2']
    # temp_vect = x['Cell_number_int_1.5']

    if type (temp_vect) == 'int':
        temp_vect = [temp_vect]
    # measure size
    d['LN_mean'] = LN_Mean(temp_vect)
    d['Geo_mean'] = Geometric_Mean(temp_vect)
    Percentile_list = list(np.percentile(temp_vect,input_percentile))
    for c,y in enumerate(input_percentile):
        temp_name = str(y)+'_percentile'
        d[temp_name] = Percentile_list[c]
    d['TTN'] = len(temp_vect) # this is total tumor number
    d['TTB'] = sum(temp_vect)
    return pd.Series(d, index=list(d.keys()))  

def LN_Mean(input_vector):
    log_vector = np.log(input_vector)
    temp_mean = log_vector.mean()
    temp_var = log_vector.var()
    if len(log_vector)==1:
        temp_var = 0 # if only one clonal
    return (math.exp(temp_mean + 0.5*temp_var))

# calculate the Geometric mean from a vector of number
def Geometric_Mean(input_vector):
    log_vector = np.log(input_vector)
    temp_mean = log_vector.mean()
    return (math.exp(temp_mean))

def fdr(p_vals):
    # FDR (Benjamini Hochberg) method
    # the P-values are first sorted and ranked. The smallest value gets rank 1, the second rank 2, and the largest gets rank N.
    # Then, each P-value is multiplied by N and divided by its assigned rank to give the adjusted P-values.
    p = np.asfarray(p_vals) # make input as float array
    by_descend = p.argsort()[::-1] # indices that give descending order
    by_orig = by_descend.argsort() # indices that give ascending order
    p = p[by_descend] # sort pvalue from small to large -> sort pvalue in descending order
    ranked_p_values = rankdata(p,method ='max') # this max is very important, when identical, use largest
    fdr = p * len(p) / ranked_p_values # fdr based on sorted and ranked pval
    fdr = np.minimum(1, np.minimum.accumulate(fdr)) # make sure the adjusted p-values do not exceed 1

    return fdr[by_orig] # return FDR-adjusted p-values in the original order

def Generate_Normalized_Metrics(input_df1,input_df2,trait_list):
    # this functional use input_df2 to normalized input_df1 using metrics defined by trait_list 
    # input_df1 is the experimental group (usually KTHC), input_df2 is the control group (usually KT)
    # trait list is TTN and TTB
    temp1 = input_df1.set_index('gRNA') # setting the index of df1 to be gRNAs
    temp2 = input_df2.set_index('gRNA').loc[temp1.index] # only use gRNAs present in the experimental group
    temp_output_df = pd.DataFrame({'gRNA':temp1.index.values}) # output df has a column named gRNA
    for temp_cname in trait_list:
        temp_cname_new = temp_cname + '_normalized' # TTN_normalized, TTB_normalized
        temp_output_df[temp_cname_new] = np.array(temp1[temp_cname].to_list())/np.array(temp2[temp_cname].to_list()) # KTHC/KT
    return(temp_output_df)

def Cal_Combined_Gene_Effect_v2(x,trait_of_interest): 
    # weighted effect of each gRNA based on TTN to see the combined gene effect
    d = {}
    temp_weight_list = x['TTN_normalized_relative'] # using TTN as the  weights associated with each gRNA
    for temp_trait in trait_of_interest: # loop through each tumor metric, e.g LN_mena, Geo_mean, etc
        # normalizes the weighted effects by dividing each term by the sum of weights
        # then calculates the sum of the normalized, weighted effects
        d[temp_trait] = sum(x[temp_trait]*temp_weight_list/sum(temp_weight_list))
    return pd.Series(d, index=list(d.keys())) 


# --------------------------------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='A function to do resampling of mice')
    parser.add_argument("--a0", required=True, help="Address of processed data of Ultra-seq, can take multiple input")
    parser.add_argument("--a1", required=False, help="Sample to exclude list address")
    parser.add_argument("--a2", required=True, help="Cell number cutoff")
    parser.add_argument("--a3", required=True, help="Number of boostrapping repeat")
    parser.add_argument("--o1", required=True, help="This the output address for summary data")
    parser.add_argument("--o2", required=False, help="This the output address for intermediate data") # df with tumor size metric of each bootstrap cycle
    parser.add_argument('--l1', nargs='+', required=False, help="A list of quantile that I want to calculate tumor size quantile: 50 60 70 80 90 95 97 99")
    parser.add_argument('--l2', nargs='+', required=False, help="A list of sgRNA sequence to exclude")
    
    # data input
    args = parser.parse_args()
    
    raw_df_input_address  = args.a0
    output_address = args.o1
    
    temp_q = [int(x) for x in args.l1]
    cell_number_cutoff = int(args.a2)
    number_of_bootstrap = int(args.a3)

    if args.l2 is None: # gRNA to exclude
        sgRNA_to_exclude = []
        print(f"No sgRNA is excluded from the analysis")
    else:
        sgRNA_to_exclude = args.l2
        print(f"sgRNAs excluded from the analysis:{sgRNA_to_exclude}")
        
    if args.a1 is None: # sample to exclude
        sample_to_exclude = []
        print(f"No sample is excluded from the analysis")
    else:
        sample_discarded_list_address = args.a1
        with open(sample_discarded_list_address, 'r') as f:
            sample_to_exclude = [line.rstrip('\n') for line in f]
        print(f"Samples excluded from the analysis:{sample_to_exclude}")
    
    # excluding LA74_14 that got only one hyg treatment
    sample_to_exclude = ['']

    #raw_summary_df = pd.read_csv(raw_df_input_address) # read input data
    raw_summary_df = pd.read_csv(raw_df_input_address) # read input data
    control_gRNA_list = Find_Controls(raw_summary_df,'safe|Neo|NT') # find the inert gRNA based on their targeted gene name
    
    # Generate bootstrapped df 
    raw_summary_df = raw_summary_df[~raw_summary_df['gRNA'].isin(sgRNA_to_exclude)] # exclude gRNA
    raw_summary_df= raw_summary_df[~raw_summary_df.Sample_ID.isin(sample_to_exclude)] # exclude the sample 
    temp_input = raw_summary_df[raw_summary_df['Identity']=='gRNA'] # consider only sgRNA but not spiekin
    
    sgRNA_number = len(temp_input[temp_input['Identity']=='gRNA']['gRNA'].unique())
    # I want to generate two name list of mice, one for experimental group and another one for control group.
    # experimental mouse group
    #cohort_1 = temp_input[temp_input['Mouse_genotype'].str.contains('KT-')]['Sample_ID'].unique() # KT
    # KTHC
    # all KTHC were treated with hyg. sgHygR decided whether the Treatment is hyg or control
    cohort_1 = temp_input[(temp_input['Mouse_genotype'].str.contains("KTHC"))] 
    #cohort_1 = cohort_1[cohort_1["Treatment"] == "control"]['Sample_ID'].unique() # control
    cohort_1 = cohort_1[cohort_1["Treatment"] == "hyg"]['Sample_ID'].unique() # hyg same no. of mice as control
    #np.random.seed(2024)
    #cohort_1 = np.random.vhoice(cohort_1, size=30, replace=False) # sample x no. of mice
    print(f"There are {len(cohort_1):d} experiment mice")

    # control mouse group
    cohort_2 = temp_input[temp_input['Mouse_genotype'].str.contains('KT-')]['Sample_ID'].unique() # KT
    print(f"There are {len(cohort_2):d} control mice")

    test_final_df = Bootstrapping_Final_df(temp_input,cohort_1,cohort_2,cell_number_cutoff,temp_q,control_gRNA_list,number_of_bootstrap,sgRNA_number)
    print(f"Bootstrapping steps have finished")
    if args.o2:
        test_final_df.to_csv(args.o2,index = False)
    else:
        print(f"No intermediate file output")
    if number_of_bootstrap!=0:
        # generate summary statistics
        temp_trait_list = ['LN_mean_relative','Geo_mean_relative','TTB_normalized_relative','TTN_normalized_relative','95_percentile_relative'] + [str(x) + '_percentile_relative' for x in temp_q]
        scaled_trait_list = [f'{x}_scaled' for x in temp_trait_list]
        temp_trait_list += scaled_trait_list
        temp_trait_list = list(set(temp_trait_list))

        Final_summary_df = Generate_Final_Summary_Dataframe(test_final_df,temp_trait_list) # gRNA level
        Final_gene_summary_df = Generate_Gene_Level_Summary_Dataframe(test_final_df,temp_trait_list)
        Final_summary_df.to_csv(output_address+'.csv',index = False)
        Final_gene_summary_df.to_csv(output_address+'_gene_level.csv',index = False)
    else:
        test_final_df.to_csv(output_address+'.csv',index = False)
    print(f"All steps finished") 

if __name__ == "__main__":
    main() 

