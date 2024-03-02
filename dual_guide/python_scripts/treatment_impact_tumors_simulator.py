import pandas as pd
import math
import numpy as np
import os
import copy
import scipy
import random
import argparse
from scipy.stats import rankdata

"""This performs level1 bootstrapping analyses.
It takes a real Ultraseq data set as input
1. sample tumors
2. reduce cell number of the sampled tumors by the desired percentage. This creates simulated treatment-sensitive tumors
    - reduce the cell number of inert tumors to create simulated treatment-sensitive tumros
    - reduce or increase the cell number of sgTS tumors to create simulated tumros with sgTS-treatment interactions
3. sample the desired no. of KTHC and KT mice in treated and untreated groups respectively
4. the simulated data sets are ready for power analysis
"""

class TreatmentImpactTumorsSimulator():
    def __init__(self, raw_data, sgRNA_to_exclude=[], sample_to_exclude=[], fraction_of_tumors=2,
                 reduction_percentage_by_treatment=None, reduction_percentage_by_treatment_TS_int=None,
                 cas9_mouse_number=None, control_mouse_number=None, **kwargs):
        """ Returns
        --------------------
        treated_cas9_indices : list of Sample_ID
        treated_control_indices : list of Sample_ID
        untreated_cas9_indices : list of Sample_ID
        untreated_control_indices : list of Sample_ID
        temp_input_untreated : df
                                untreated mouse group
        temp_input_treatment_specific : df
                                        treatment mouse group

        Parameters
        --------------------
        reduction_percentage_by_treatment (float): a float representing how much treatment affect cell number of each tumor
        reduction_percentage_by_treatment_TS_int (float): genotype-environment interaction term
        cas9_mouse_number (int): no. of KTHC mice to generate a simulated KTHC cohort
        control_mouse_number (int): no. of KT mice to generate a simulated control (KT) cohort
        """
        self.raw_data = kwargs.get('raw_data', raw_data)
        self.data = self._load_data()
        self.sgRNA_to_exclude = kwargs.get('sgRNA_to_exclude',sgRNA_to_exclude)
        self.sample_to_exclude = kwargs.get('sample_to_exclude', sample_to_exclude)
        self.fraction_of_tumors = kwargs.get('fraction_of_tumors', fraction_of_tumors)
        self.reduction_percentage_by_treatment = float(kwargs.get('reduction_percentage_by_treatment', reduction_percentage_by_treatment))
        self.reduction_percentage_by_treatment_TS_int = float(kwargs.get('reduction_percentage_by_treatment_TS_int',reduction_percentage_by_treatment_TS_int))
        self.cas9_mouse_number = kwargs.get('cas9_mouse_number', cas9_mouse_number)
        self.control_mouse_number = kwargs.get('control_mouse_number', control_mouse_number)
        # self.trait_list = ['LN_mean_relative', 'Geo_mean_relative', '95_percentile_relative',
        #                    'TTB_normalized_relative', 'TTN_normalized_relative', 'TTN']
        self.sgRNA_number = None
        self.temp_input_treatment_specific = None
        self.temp_input_untreated = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _load_data(self):
        # Check if self.raw_data is a DataFrame or a directory
        if isinstance(self.raw_data, pd.DataFrame):
            # If it's a DataFrame, use it directly
            self.data = self.raw_data.copy(deep=True)
        elif isinstance(self.raw_data, str) and os.path.isfile(self.raw_data):
            # If it's a string and a directory, load data from files in the directory
            self.data = pd.read_csv(self.raw_data).copy(deep=True)
        else:
            raise ValueError("Invalid input for self.raw_data. Should be a DataFrame or a valid file.")
        return self.data

    def preprocess_data(self): # exclude Sample_ID and ans sgRNA if you want to exclude some of them
        if self.data is not None:
            self.data = self.data[~self.data['gRNA'].isin(self.sgRNA_to_exclude)]
            self.data = self.data[~self.data['Sample_ID'].isin(self.sample_to_exclude)]
            # only look at sgRNAs not spike-ins
            self.data = self.data[self.data['Identity'] == 'gRNA']
            self.sgRNA_number = len(self.data[self.data['Identity'] == 'gRNA']['gRNA'].unique())
            print(f"Number of unique sgRNA: {self.sgRNA_number}")
            #return self.data
        else:
            print("Data not loaded. Call load_data method first.")  
    
    def _group_mice_and_bootstrap_tumors(self): 
        # level 1 bootstrapping: bootstrap tumors to create treatment-sensitive tumors
        # group mice based on genotypes: KTHC (experimental) vs KT (untreated) mice
        temp_input_KT = self.data[self.data.Mouse_genotype.str.contains('KT-')]
        temp_input_KTHC = self.data[self.data.Mouse_genotype.str.contains('KTHC')]

        # Bootstrap and process tumors for KTHC mice
        # this creates simulated treatment-sensitive tumors
        treatment_specific_KTHC_inert = self._bootstrap_and_process_tumors(temp_input_KTHC, 'Inert')
        treatment_specific_KTHC_TS = self._bootstrap_and_process_tumors(temp_input_KTHC, 'TS')

        # Combine data for treatment condition. This is the treatment-specific df we're going to run power analysis on
        # ------------------------------------
        temp_input_treatment_specific = pd.concat([temp_input_KT, treatment_specific_KTHC_inert, treatment_specific_KTHC_TS])
        # Create a subset of untreated KTHC mice. This is the untreated df we're going to run and power analysis on and compare the treatment-specific df to
        temp_input_KTHC_untreated = temp_input_KTHC.query("~index.isin(@treatment_specific_KTHC_inert.index) & ~index.isin(@treatment_specific_KTHC_TS.index)")
        #temp_input_KTHC_untreated = temp_input_KTHC.loc[(~temp_input_KTHC.index.isin(treatment_specific_KTHC_inert.index.values)) & (~temp_input_KTHC.index.isin(treatment_specific_KTHC_TS.index.values))]
        # Combine data for untreated mice. This is the untreated df we're going to run power analysis on
        # ----------------------------------
        temp_input_untreated = pd.concat([temp_input_KT, temp_input_KTHC_untreated])
        # store treatment specific and untreated as instance attributes
        self.temp_input_treatment_specific, self.temp_input_untreated = temp_input_treatment_specific, temp_input_untreated
        print("Treatment specific data")
        self.temp_input_treatment_specific.head()
        # print(self.temp_input_treatment_specific.to_string(index=False))
        #display(self.temp_input_treatment_specific.head())
        print("\nUntreated Data:")
        # print(self.temp_input_untreated.to_string(index=False))
        self.temp_input_untreated.head()
        #display(self.temp_input_untreated.head())

    def get_mouse_indices_per_treatment(self):
        # Call _group_mice_and_bootstrap_tumors using self
        self._group_mice_and_bootstrap_tumors()
         # Check if the attributes are not None before proceeding
        if self.temp_input_treatment_specific is None or self.temp_input_untreated is None:
            raise ValueError("Call _group_mice_and_bootstrap_tumors first.")
        # cohort1 for treated
        treated_cas9_indices = self._bootstrap_mouse_index('KTHC', 'treated')
        #treated_control_indices = self._bootstrap_mouse_index('KT-', 'treated') # not boostrap KT
        treated_control_indices = self.temp_input_treatment_specific[self.temp_input_treatment_specific['Mouse_genotype'].str.contains('KT-')]['Sample_ID'].unique()
        # cohort2 for untreated
        untreated_cas9_indices = self._bootstrap_mouse_index('KTHC', 'untreated')
        #untreated_control_indices = self._bootstrap_mouse_index('KT-', 'untreated') # not boostrap KT
        untreated_control_indices = self.temp_input_untreated[self.temp_input_untreated['Mouse_genotype'].str.contains('KT-')]['Sample_ID'].unique()
        print(f'cas9 mice in treatment group are {treated_cas9_indices}')
        print(f'control mice in treatment group are {treated_control_indices}')
        print(f'cas9 in untreated group are {untreated_cas9_indices}')
        print(f'control mice in untreated gorup are {untreated_control_indices}')
        return treated_cas9_indices, treated_control_indices, untreated_cas9_indices, untreated_control_indices
    
    def _bootstrap_mouse_index(self, genotype, treatment):
        input_df = self.temp_input_treatment_specific if treatment == 'treated' else self.temp_input_untreated
        mouse_indices = input_df[input_df['Mouse_genotype'].str.contains(genotype)]['Sample_ID'].unique()
        sample_size = self.cas9_mouse_number if genotype == 'KTHC' else self.control_mouse_number
        mouse_indices_bootstrapped = np.random.choice(mouse_indices, size=sample_size, replace=True)
        return mouse_indices_bootstrapped
    
    def _bootstrap_and_process_tumors(self, input_df, gene): # simulated hyg treated tumors
        temp_input_subset = self._subset_tumors(input_df, gene)
        resampled_indices = self._bootstrap_tumors(temp_input_subset.index.values)
        treatment_specific_subset = self._reduce_cell_number(temp_input_subset, resampled_indices, gene)
        return treatment_specific_subset
    
    def _subset_tumors(self, input_df, gene):
        if gene == 'Inert':
            input_subset = input_df[input_df['Targeted_gene_name'] == "Inert"]
        elif gene == 'TS':
            input_subset = input_df[(input_df['Targeted_gene_name'] != "Pcna") & (input_df['Targeted_gene_name'] != 'Inert')]
        return input_subset
    
    def _bootstrap_tumors(self, indices):
        resampled_tumor_indices = np.random.choice(indices, size=len(indices)//self.fraction_of_tumors, replace=True)
        return resampled_tumor_indices

    def _reduce_cell_number(self, input_df, indices, gene):
        if gene == 'Inert':
            input_df.loc[indices, 'Cell_number'] = input_df.loc[indices, 'Cell_number'] * (1 - self.reduction_percentage_by_treatment)
        elif gene == 'TS':
            input_df.loc[indices, 'Cell_number'] = input_df.loc[indices, 'Cell_number'] * (1 - self.reduction_percentage_by_treatment_TS_int)
        return input_df.loc[indices]


    
