import os
import pandas as pd
import numpy as np
import copy
from scipy.stats import rankdata
#from Power_analysis_by_bootstrapping2 import *
from UltraSeq_bootstrapping import *
from UltraSeq_metrics_functions import *

class PowerAnalysisTreatmentEffect():
    DEFAULT_TRAIT_LIST = ['LN_mean_relative', 'Geo_mean_relative',
                         '95_percentile_relative', 'TTB_normalized_relative',
                         'TTN_normalized_relative', 'TTN']
    DEFAULT_CELL_NUMBER_CUTOFF = 300
    DEFAULT_PERCENTILE_LIST = [50, 60, 70, 80, 90, 95, 97, 99]
    DEFAULT_NUMBER_MOUSE_BOOTSTRAP = 10
    DEFAULT_NUMBER_GRNA_BOOTSTRAP = 1000
    #CONTROL_GRNA_PATTERN='safe|Neo|NT'

    def __init__(self, raw_treatment_specific_df, raw_untreated_df,
                 treatment_specific_cas9_sample_list=None, treatment_specific_control_sample_list=None,
                 untreated_cas9_sample_list=None, untreated_control_sample_list=None,
                 trait_list=None, cell_number_cutoff=None, input_control_gRNA_list=None,
                 control_gRNA_pattern='Inert', mouse_bootstrap_repeats=None, gRNA_bootstrap_repeats=None,
                 input_total_gRNA_number=None,
                 percentile_list=None, input_factor=1, **kwargs):
        """
        Power analysis for treatment effect
        Parameters
        ----------
        treatment_specific_cas9_sample_list (list): KTHC mice in treatment group
        treatment_specific_control_sample_list (list): KT (control) mice in treatment grouopf
        untreated_cas9_sample_list (list): KTHC mice in untreated group
        untreated_control_sample_list (list): KT mice in untreated group
        percentile_list (list) : some no. separate
        mouse_bootstrap_repeats (int): within cohort bootstrapping. level2 mice bootstrapping. 
                                        This happens after sampling tumors to generate simulated treatment-sensitive tumors
                                        and sampling mice to generate KTHC cohort (level1 bootstrapping).
        gRNA_bootstrap_repeats (int): is within mouse gRNA resampling level3 gRNA bootstrapping
        to pass in parameters:
        # Create an instance of the class with specific parameter values
        # use list
        obj = PowerAnalysisTreatmentEffect(treatment_specific_cas9_sample_list=[1, 2, 3],
                            untreated_cas9_sample_list=[4, 5, 6],
                            custom_param='some_value')
        # use dic
        parameters = {'treatment_specific_cas9_sample_list': [1, 2, 3],
                        'untreated_cas9_sample_list': [4, 5, 6],
                        'custom_param': 'some_value',}
        # Create an instance of YourClassName and pass the dictionary as **kwargs
        obj = PowerAnalysisTreatmentEffect(**parameters)                
        """
        self.raw_treatment_specific_df = kwargs.get('raw_treatment_specific_df', raw_treatment_specific_df)
        self.raw_untreated_df = kwargs.get('raw_untreated_df', raw_untreated_df)
        self.cell_number_cutoff = int(kwargs.get('cell_number_cutoff', cell_number_cutoff)) if cell_number_cutoff is not None else self.DEFAULT_CELL_NUMBER_CUTOFF
        # load treatment df and remove small tumors
        self.treatment_specific_df = self._load_data(self.raw_treatment_specific_df)
        # load untreated df and remove small tumors
        self.untreated_df = self._load_data(self.raw_untreated_df)
        # get mouse list in each group
        self.treatment_specific_cas9_sample_list = kwargs.get('treatment_specific_cas9_sample_list', treatment_specific_cas9_sample_list)
        self.treatment_specific_control_sample_list = kwargs.get('treatment_specific_control_sample_list', treatment_specific_control_sample_list)
        self.untreated_cas9_sample_list = kwargs.get('untreated_cas9_sample_list', untreated_cas9_sample_list)
        self.untreated_control_sample_list = kwargs.get('untreated_control_sample_list', untreated_control_sample_list)
        # default values
        self.trait_list = kwargs.get('trait_list', trait_list) if trait_list is not None else self.DEFAULT_TRAIT_LIST
        # bootstrap repeats and congtrol gRNA list
        self.input_control_gRNA_list = kwargs.get('input_control_gRNA_list', input_control_gRNA_list)
        self.mouse_bootstrap_repeats = kwargs.get('mouse_bootstrap_repeats', mouse_bootstrap_repeats)
        self.gRNA_bootstrap_repeats = kwargs.get('gRNA_bootstrap_repeats', gRNA_bootstrap_repeats)
        self.input_total_gRNA_number = kwargs.get('input_total_gRNA_number', input_total_gRNA_number)
        self.control_gRNA_pattern = kwargs.get('control_gRNA_pattern', control_gRNA_pattern)
        #self.percentile_list = [int(p) for p in kwargs.get('percentile_list', str(percentile_list)).split()] if percentile_list is not None else self.DEFAULT_PERCENTILE_LIST
        self.percentile_list = kwargs.get('percentile_list', percentile_list) if percentile_list is not None else self.DEFAULT_PERCENTILE_LIST
        self.input_factor = float(kwargs.get('input_factor', input_factor))
        self.treatment_specific_mouse_index_dic = None
        self.untreated_mouse_index_dic = None
        self.scaled_trait_list = [f"{x}_scaled" for x in self.trait_list]
        # Handle any additional keyword arguments (kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _load_data(self, input_df):
        # Check if input_df is a DataFrame or a directory
        if isinstance(input_df, pd.DataFrame):
            # If it's a DataFrame, use it directly
            input_df = input_df.copy(deep=True)
        elif isinstance(input_df, str) and os.path.isfile(input_df):
            # If it's a string and a directory, load data from files in the directory
            input_df = pd.read_csv(input_df).copy(deep=True)
        else:
            raise ValueError("Invalid input for input_df. Should be a DataFrame or a valid file.")
        input_df = self._remove_small_tumors(input_df) # removing small tumors
        return input_df
    
    def _remove_small_tumors(self, input_df):
        input_df = input_df[input_df['Cell_number'] > self.cell_number_cutoff]
        return input_df
    
    def find_control(self):
        control_gRNA_list = self.treatment_specific_df.loc[self.treatment_specific_df['Targeted_gene_name'].str.contains(self.control_gRNA_pattern, na=False, regex=True),'gRNA'].unique()
        self.input_control_gRNA_list = control_gRNA_list
    
    def mouse_number_sgRNA_abundance_power_analysis_treatment_effect(self):
        self.treatment_specific_mouse_index_dic = self.Generate_Index_Dictionary(treatment='treated')
        self.untreated_mouse_index_dic = self.Generate_Index_Dictionary(treatment='untreated')
        for i in range(self.mouse_bootstrap_repeats): # level2 within simulated cohort mosue bootstrapping
            # I only did level2 within simulated cohort bs for KTHC mice
            treatment_cas9_resamples_list = self._resample_mice(self.treatment_specific_cas9_sample_list)
            # treatment_control_resamples_list = self._resample_mice(self.treatment_specific_control_sample_list)
            untreated_cas9_resamples_list = self._resample_mice(self.untreated_cas9_sample_list)
            # untreated_control_resamples_list = self._resample_mice(self.untreated_control_sample_list)
            # test_final_df = self.bootstrapping_final_df_treatment_impact(treatment_cas9_resamples_list, treatment_control_resamples_list,
            #                                           untreated_cas9_resamples_list, untreated_control_resamples_list)
            test_final_df = self.bootstrapping_final_df_treatment_impact(treatment_cas9_resamples_list, self.treatment_specific_control_sample_list,
                                                                        untreated_cas9_resamples_list, self.untreated_control_sample_list)
            # summarize sgRNA level result
            temp_final_summary_df = Generate_Final_Summary_Dataframe(test_final_df,self.scaled_trait_list)
            #temp_final_summary_df['Level2_bootstrap_id'] = 'Size_'+str(len(treatment_cas9_resamples_list)) + '_Rep' +str(i)
            temp_final_summary_df['Level2_bootstrap_id'] = 'Size_'+str(len(self.treatment_specific_cas9_sample_list)) + '_Rep' +str(i)
             # summarize gene level result
            temp_gene_summary_df = Generate_Gene_Level_Summary_Dataframe(test_final_df,self.scaled_trait_list)
            #temp_gene_summary_df['Level1_bootstrap_id'] = 'Size_'+str(len(treatment_cas9_resamples_list)) + '_Rep' +str(i)
            temp_gene_summary_df['Level2_bootstrap_id'] = 'Size_'+str(len(self.treatment_specific_cas9_sample_list)) + '_Rep' +str(i)

            if i == 0:
                temp_output_df = temp_final_summary_df.reset_index(drop = True)
                temp_output_df2 = temp_gene_summary_df.reset_index(drop = True)
            else:
                temp_output_df = pd.concat([temp_output_df,temp_final_summary_df.reset_index(drop = True)])
                temp_output_df2 = pd.concat([temp_output_df2,temp_gene_summary_df.reset_index(drop = True)])
        return temp_output_df,temp_output_df2
    
    def Generate_Index_Dictionary(self, treatment):
        # This function generate a dictionary for speed up the boostrap process
        # The key for the dictionary is the sample(mice) id and the value is a list of all the index for tumors in this mice.
        temp_dic = {}
        input_df = self.treatment_specific_df if treatment == 'treated' else self.untreated_df
        temp_group = input_df.groupby(['Sample_ID'])
        for key in temp_group.groups.keys():
            temp_dic[key] = temp_group.get_group(key).index.values
        return temp_dic   
    
    def _resample_mice(self, input_df):
        sample_list = np.random.choice(input_df, size=len(input_df), replace=True)
        return sample_list
    
    def bootstrapping_final_df_treatment_impact(self, treatment_cas9_resamples_list, treatment_control_resamples_list,
                                                    untreated_cas9_resamples_list, untreated_control_resamples_list):
    #def bootstrapping_final_df_treatment_impact(self):
        # only want to stores these values in the final df
        selected_cols = ['Level3_bootstrap_id', 'gRNA', 'Targeted_gene_name', 'TTN', 'TTN_normalized_relative'] + self.scaled_trait_list 

        for i in range(self.gRNA_bootstrap_repeats):
            # this step is do a two level (level2&3) bootstrapping to generate a list of index -> each index map to a row of the tumor info dataframe
            # level2 bs: resample mice
            # level3 bs: resample gRNA
            # treatment group
            # KTHC
            #x = Nested_Boostrap_Index_single(self.treatment_specific_mouse_index_dic, self.treatment_specific_cas9_sample_list, self.input_factor)
            x = Nested_Boostrap_Index_single(self.treatment_specific_mouse_index_dic, treatment_cas9_resamples_list, self.input_factor)
            temp_treatment_bootstrap_cas9_df = self.treatment_specific_df.loc[x]
            # # KT
            y = Nested_Boostrap_Index_Special_single(self.treatment_specific_mouse_index_dic, self.treatment_specific_df, self.input_total_gRNA_number, 
                                                     treatment_control_resamples_list, self.input_factor)
            # y = Nested_Boostrap_Index_Special_single(self.treatment_specific_mouse_index_dic, self.treatment_specific_df, self.input_total_gRNA_number,
            #                                          self.treatment_specific_control_sample_list, self.input_factor)
            temp_treatment_bootstrap_control_df= self.treatment_specific_df.loc[y]
            # # untreated group
            #w = Nested_Boostrap_Index_single(self.untreated_mouse_index_dic, self.untreated_cas9_sample_list, self.input_factor)
            w = Nested_Boostrap_Index_single(self.untreated_mouse_index_dic, untreated_cas9_resamples_list, self.input_factor)
            temp_untreated_bootstrap_cas9_df = self.untreated_df.loc[w]
            z = Nested_Boostrap_Index_Special_single(self.untreated_mouse_index_dic, self.untreated_df, self.input_total_gRNA_number,
                                                     untreated_control_resamples_list, self.input_factor)
            # z = Nested_Boostrap_Index_Special_single(self.untreated_mouse_index_dic, self.untreated_df, self.input_total_gRNA_number,
            #                                          self.untreated_control_sample_list, self.input_factor)
            temp_untreated_bootstrap_control_df = self.untreated_df.loc[z]

            temp_metric_treatment_df = Calculate_Relative_Normalized_Metrics(temp_treatment_bootstrap_cas9_df,temp_treatment_bootstrap_control_df,
                                                                             self.percentile_list, self.input_control_gRNA_list)
            temp_metric_untreated_df = Calculate_Relative_Normalized_Metrics(temp_untreated_bootstrap_cas9_df, temp_untreated_bootstrap_control_df,
                                                                             self.percentile_list, self.input_control_gRNA_list)
            # append the relative scaled metrics to treatment_df
            self.add_cohort_specific_relative_metrics_scaled_to_untreated(temp_metric_treatment_df, temp_metric_untreated_df)
            # Add_Corhort_Specific_Relative_Metrics(temp_metric_treatment_df, self.input_control_gRNA_list)
            # Add_Corhort_Specific_Relative_Metrics(temp_metric_untreated_df, self.input_control_gRNA_list)
            temp_metric_treatment_df['Level3_bootstrap_id'] = ['B'+str(i)]*temp_metric_treatment_df.shape[0] # add bootstrap id colums to treatment_df
            if i == 0:
                temp_out_df = temp_metric_treatment_df.loc[:, selected_cols]
            else:
                temp_out_df = pd.concat([temp_out_df.reset_index(drop = 'True'),temp_metric_treatment_df.reset_index(drop = 'True')]) # append each bootstrap to end of temp_out_df
                temp_out_df = temp_out_df.loc[:, selected_cols]
        return temp_out_df
        
    def add_cohort_specific_relative_metrics_scaled_to_untreated(self, treatment_specific_df, untreated_df):
        # append the relative scaled metrics to untreated_df
        # scale sgHygR-sgTS/median(sgHygR-sgInert) (treatment) to sgTS-sgInert/median(sgInert-sgInert) (untreated)
        # scaled all relative metrics in treatment to untreated
        # treatment_specific_df is the treatment group
        # untreated_df is the untreated group 
        for metric in self.trait_list:
            treatment_specific_df[f'{metric}_scaled'] = treatment_specific_df[metric] / untreated_df[metric]
            #untreated_df[f'{metric}_scaled'] = untreated_df[metric] / untreated_df[metric]


    # def Add_Corhort_Specific_Relative_Metrics(input_df,input_control_list):
    #     tumor_metrics = ['']
    #     # Add relative metrics for LN_mean, GEO_mean etc using the median of inert
    #     temp_sub = input_df[input_df['gRNA'].isin(input_control_list)]
    #     #for temp_cname in input_df.drop(columns=['gRNA'],inplace = False).columns:
    #     trait_of_interest = ['LN_mean', 'Geo_mean', '95_percentile', 'TTB_normalized', 'TTN_normalized', 'TTN']
    #     for temp_cname in trait_of_interest:
    #         temp_name = temp_cname+'_relative'
    #         input_df[temp_name] = input_df[temp_cname]/temp_sub[temp_cname].median()