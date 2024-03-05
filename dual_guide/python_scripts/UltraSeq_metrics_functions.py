import pandas as pd
import numpy as np
import math
from scipy.stats import rankdata

''' This module contains functions to calcualte tumor metrics, including LN mean, percentiles, TTN (total tumor number), TTB (total tumor burden). 
    It also contains functions to normalize TTN and TTB to KT mice and calculate metrics relative to inert tumors. 
    If looking at treatment effect, it compares normalized relative and relative metrics in the treatment to the untreated group
'''

def Cal_Combined_Gene_Effect_v2(x,trait_of_interest):
    d = {}
    temp_weight_list = x['TTN_normalized_relative']
    for temp_trait in trait_of_interest:
        d[temp_trait] = sum(x[temp_trait]*temp_weight_list/sum(temp_weight_list)) # weighted sum by normalized TTN relative to inert
    return pd.Series(d, index=list(d.keys())) 

def Calculate_Relative_Normalized_Metrics(input_df1,input_df2,percentile_list,input_control_gRNA_list):
    # Cas9 mouse
    # print('haha'+str(input_df1.shape[0]))
    temp_df = input_df1.reset_index().groupby(['gRNA'],as_index = False).apply(Cal_Tumor_Size_simple,(percentile_list))
    # print(input_df1[['gRNA','Cell_number']])
    # print(temp_df)
    # Control mouse
    temp_df2 = input_df2.reset_index().groupby(['gRNA'],as_index = False).apply(Cal_Tumor_Size_simple,(percentile_list))
    
    # normalize TTN and TTB to KT
    temp_out = Generate_Normalized_Metrics(temp_df,temp_df2,['TTN','TTB'])
    temp_df = temp_df.merge(temp_out,on ='gRNA') # merge

    # calculate relative expression to sgInert
    Add_Corhort_Specific_Relative_Metrics(temp_df,input_control_gRNA_list)

    # annotate sample type
    temp_df['Type'] = temp_df.apply(lambda x: 'Inert' if (x['gRNA'] in input_control_gRNA_list) else 'Experiment',axis=1)
    #temp_df = temp_df.merge(input_df1[['gRNA','Targeted_gene_name',
    #  'Identity', 'Numbered_gene_name']].drop_duplicates(),how = 'inner',on = 'gRNA')
    temp_df = temp_df.merge(input_df1[['gRNA','Targeted_gene_name',
    'Identity', 'leftGuide', 'rightGuide', 'leftRightGuide']].drop_duplicates(),how = 'inner',on = 'gRNA')
    return(temp_df)


def Cal_Bootstrapping_Summary(x,trait_of_interest):    
    d = {}
    for temp_trait in trait_of_interest:
        # temp0 = temp_trait + '_95P'
        # temp1 = temp_trait + '_5P'
        temp2 = temp_trait +'_fraction_greater_than_one' # t_test pvalue column name
        temp3 = temp_trait +'_bootstrap_median'
        temp4 = temp_trait +'_bootstrap_mean'
        temp5 = temp_trait + '_97.5P'
        temp6 = temp_trait + '_2.5P'
        # d[temp0] = x[temp_trait].quantile(0.95)
        # d[temp1] = x[temp_trait].quantile(0.05)
        d[temp2] = sum(x[temp_trait]>1)/len(x[temp_trait])
        d[temp3] = x[temp_trait].mean()
        d[temp4] = x[temp_trait].median()
        d[temp5] = x[temp_trait].quantile(0.975)
        d[temp6] = x[temp_trait].quantile(0.025)
    d['TTN_bootstrap_median'] = x['TTN'].median()
    return pd.Series(d, index=list(d.keys())) 

def Cal_Bootstrapping_Summary_V2(x,trait_of_interest):    
    d = {}
    for temp_trait in trait_of_interest:
        # temp0 = temp_trait + '_95P'
        # temp1 = temp_trait + '_5P'
        temp2 = temp_trait +'_fraction_greater_than_one' # t_test pvalue column name
        temp3 = temp_trait +'_bootstrap_median'
        temp4 = temp_trait +'_bootstrap_mean'
        temp5 = temp_trait + '_97.5P'
        temp6 = temp_trait + '_2.5P'
        # d[temp0] = x[temp_trait].quantile(0.95)
        # d[temp1] = x[temp_trait].quantile(0.05)
        d[temp2] = sum(x[temp_trait]>1)/len(x[temp_trait])
        d[temp3] = x[temp_trait].mean()
        d[temp4] = x[temp_trait].median()
        d[temp5] = x[temp_trait].quantile(0.975)
        d[temp6] = x[temp_trait].quantile(0.025)
    # d['TTN_bootstrap_median'] = x['TTN'].median()
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
    p = np.asfarray(p_vals) # make input as float array
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    p = p[by_descend] # sort pvalue from small to large
    ranked_p_values = rankdata(p,method ='max') # this max is very important, when identical, use largest
    fdr = p * len(p) / ranked_p_values
    fdr = np.minimum(1, np.minimum.accumulate(fdr))

    return fdr[by_orig]

# calculate the tumor size using 3 measurements for each sgRNA
def Cal_Tumor_Size_simple(x,input_percentile):
    d = {}
    temp_vect = x['Cell_number']
    if type (temp_vect) == 'int':
        temp_vect = [temp_vect]
    # measure size
    d['LN_mean'] = LN_Mean(temp_vect)
    d['Geo_mean'] = Geometric_Mean(temp_vect)
    Percentile_list = list(np.percentile(temp_vect,input_percentile))
    for c,y in enumerate(input_percentile):
        temp_name = str(y)+'_percentile'
        d[temp_name] = Percentile_list[c]
    # measure diversity
    # print('start:'+'\n')
    # print(temp_vect)
    # print('end:'+'\n')
    d['TTN'] = len(temp_vect) # this is total tumor number
    d['TTB'] = sum(temp_vect)
    return pd.Series(d, index=list(d.keys())) 


def Generate_Normalized_Metrics(input_df1,input_df2,trait_list):
    # this functional use input_df2 to normalized input_df1 using metrics defined by trait_list 
    # input_df1 is the experimental group (KTHC), input_df2 is the control group (KT)
    temp1 = input_df1.set_index('gRNA')
    temp2 = input_df2.set_index('gRNA').loc[temp1.index]
    temp_output_df = pd.DataFrame({'gRNA':temp1.index.values})
    for temp_cname in trait_list:
        temp_cname_new = temp_cname + '_normalized'
        temp_output_df[temp_cname_new] = np.array(temp1[temp_cname].to_list())/np.array(temp2[temp_cname].to_list())
    return(temp_output_df)

def Add_Corhort_Specific_Relative_Metrics(input_df,input_control_list):
    # Add relative metrics for LN_mean, GEO_mean etc using the median of inert
    temp_sub = input_df[input_df['gRNA'].isin(input_control_list)]
    #for temp_cname in input_df.drop(columns=['gRNA'],inplace = False).columns:
    trait_of_interst = ['LN_mean', 'Geo_mean', '95_percentile', 'TTB_normalized', 'TTN_normalized', 'TTN']
    for temp_cname in trait_of_interst:
        temp_name = temp_cname+'_relative'
        input_df[temp_name] = input_df[temp_cname]/temp_sub[temp_cname].median()

# def add_cohort_specific_relative_metrics_scaled_to_untreated(treatment_specific_df, untreated_df, relative_metrics):
#     # append the relative metrics scaled to untreated
#     # scale treatment-sgTS/median of treatment-sgInert (sgHygR-sgInert hyg treatment) to sgTS-sgInert/median of sgInert-sgInert (untreated)
#     # scaled all relative metrics in treatment to untreated group
#     # treatment_specific_df is the treatment group
#     # untreated_df is the untreated group
#     for metric in relative_metrics:
#         treatment_specific_df[f'{metric}_scaled'] = treatment_specific_df[metric] / untreated_df[metric] # only interested in treatment/untreated
#         #untreated_df[f'{metric}_scaled'] = untreated_df[metric] / untreated_df[metric]
