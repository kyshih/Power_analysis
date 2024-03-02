from treatment_impact_tumors_simulator import TreatmentImpactTumorsSimulator
from power_analysis_treatment_effect import PowerAnalysisTreatmentEffect
import argparse

# from Power_analysis_by_bootstrapping2 import *

''' This creates a TreatmentImpactTumorsSimulator object to generate simulated hyg tumors and simulated KT and KTHC cohorts. It also creates treatment-specific df and untreated df containing
    all the tumors. This is level1 bootstrapping. It then pass the indices of the simulated cohorts and the two dfs to the PowerAnalysisTreatmentEffect object. 
    This is level2 within simulated cohort bootstrapping and level3 mouse and gRNA bootstrapping.
'''

def main():
    parser = argparse.ArgumentParser(description='A function to do resampling of mice')
    parser.add_argument("--bootstrap_level1_cycle", required=True, help="level1 bootstrap repeat no.")
    parser.add_argument("--processed_ultraseq_data", required=True, help="Address of processed data of Ultra-seq")
    parser.add_argument("--sample_to_exclude_list", required=False, help="Sample to exclude list address")
    parser.add_argument("--cell_number_cutoff", required=False, help="Cell number cutoff")
    parser.add_argument("--cell_number_reduction_precentage_by_treatment", required=True, help="cell number reduction percentage by treatment")
    parser.add_argument("--cell_number_reduction_percentage_by_treatment_TS_interaction", required=True, help="cell number reduction percentage by treatment and sgTS interaction")
    parser.add_argument("--fraction_of_tumors_to_create_treated_tumors", required=False, help="fraction of tumors to create simulate treated tumors")
    parser.add_argument("--cas9_mouse_number", required=True, help="This is the cas9 mouse number I want to bootstrap")
    parser.add_argument("--control_mouse_number", required=True, help="This is the KT (control) mouse number I want to bootstrap")
    parser.add_argument("--mouse_bootstrap_repeats", required=True, help="This is the number of mouse bootstrap repeats")
    parser.add_argument("--gRNA_bootstrap_repeats", required=True, help="This is the number of gRNA bootstrap repeats")
    parser.add_argument("--control_gRNA_pattern", required=False, help="pattern to identify control tumors")
    parser.add_argument("--final_output_address", required=True, help="This the output address for summary data")
    parser.add_argument("--intermediate_output_address", required=False, help="This the output address for intermediate data") # df with tumor size metric of each bootstrap cycle
    parser.add_argument('--percentile_list', nargs='+', required=False, help="A list of quantile that I want to calculate tumor size quantile: 50 60 70 80 90 95 97 99")
    parser.add_argument('--gRNA_sequence_to_exclude_list', nargs='+', required=False, help="A list of sgRNA sequence to exclude")

    # data input
    args = parser.parse_args()
    
    bootstrap_level1_cycle = str(args.bootstrap_level1_cycle)
    raw_data_address = args.processed_ultraseq_data

    output_address = args.final_output_address
    
    reduction_percentage_by_hyg = float(args.cell_number_reduction_precentage_by_treatment)
    reduction_percentage_by_hyg_TS_int = float(args.cell_number_reduction_percentage_by_treatment_TS_interaction)
    input_cas9_mouse_number = int(args.cas9_mouse_number)
    input_control_mouse_number = int(args.control_mouse_number)
    input_mouse_bootstrap_repeats = int(args.mouse_bootstrap_repeats)
    input_gRNA_bootstrap_repeats= int(args.gRNA_bootstrap_repeats)

    if args.percentile_list is not None:
        input_percentile_list = [int(x) for x in args.percentile_list]
    if args.control_gRNA_pattern is not None:
        input_control_gRNA_pattern = str(args.control_gRNA_pattern)
    if args.cell_number_cutoff is not None:
        input_cell_number_cutoff = int(args.cell_number_cutoff)
    if args.fraction_of_tumors_to_create_treated_tumors is not None:
        fraction_of_tumors = int(args.fraction_of_tumors_to_create_treated_tumors)
    if args.gRNA_sequence_to_exclude_list is None: # gRNA to exclude
        sgRNA_to_exclude = []
        print(f"No sgRNA is excluded from the analysis")
    else:
        sgRNA_to_exclude = args.gRNA_sequence_to_exclude_list
        print(f"sgRNAs excluded from the analysis:{sgRNA_to_exclude}")
        
    if args.sample_to_exclude_list is None: # sample to exclude
        sample_to_exclude = []
        print(f"No sample is excluded from the analysis")
    else:
        sample_discarded_list_address = args.sample_to_exclude_list
        with open(sample_discarded_list_address, 'r') as f:
            sample_to_exclude = [line.rstrip('\n') for line in f]
        print(f"Samples excluded from the analysis:{sample_to_exclude}")

    print('Running simulation of hyg treated tumors')
    print('This is level1 bootstrapping. A fraction of tumors are sampled to be hyg-sensitive')
    print('------------------------------------------------------------------------------------------------')
    simulated_hyg_tumors = TreatmentImpactTumorsSimulator(raw_data=raw_data_address,
                                                                reduction_percentage_by_treatment=reduction_percentage_by_hyg,
                                                                reduction_percentage_by_treatment_TS_int=reduction_percentage_by_hyg_TS_int,
                                                                cas9_mouse_number=input_cas9_mouse_number,
                                                                control_mouse_number=input_control_mouse_number)
    simulated_hyg_tumors.preprocess_data()
    treated_cas9_indices, treated_control_indices, untreated_cas9_indices, untreated_control_indices = simulated_hyg_tumors.get_mouse_indices_per_treatment()
    print('Simulation of hyg treated tumors has finished.')
    print('------------------------------------------------------------------------------------------------')
    print('Running power analysis of hyg treatment effect on sgTS- and sgInert-targeted tumors')
    power_analysis_hyg_effect = PowerAnalysisTreatmentEffect(raw_treatment_specific_df=simulated_hyg_tumors.temp_input_treatment_specific,
                                                            raw_untreated_df=simulated_hyg_tumors.temp_input_untreated,
                                                            treatment_specific_cas9_sample_list=treated_cas9_indices,
                                                            treatment_specific_control_sample_list=treated_control_indices,
                                                            untreated_cas9_sample_list=untreated_cas9_indices,
                                                            untreated_control_sample_list=untreated_control_indices,
                                                            percentile_list=input_percentile_list, input_total_gRNA_number=simulated_hyg_tumors.sgRNA_number,
                                                            mouse_bootstrap_repeats=input_mouse_bootstrap_repeats, gRNA_bootstrap_repeats=input_gRNA_bootstrap_repeats)
    power_analysis_hyg_effect.find_control()
    print(f"The control gRNAs in this simulation are {power_analysis_hyg_effect.input_control_gRNA_list}")
    print(f"There are {len(power_analysis_hyg_effect.input_control_gRNA_list)} control gRNAs total in this simulation")
    print('-------------------------------------------------------------------------------------------------')
    result_gRNA_summary_df, result_gene_summary_df = power_analysis_hyg_effect.mouse_number_sgRNA_abundance_power_analysis_treatment_effect()
    # append level1 bootstrap cycle no.
    result_gRNA_summary_df['Level1_bootstrap_id'] = 'Size_'+ str(input_cas9_mouse_number) + '_Rep' + bootstrap_level1_cycle
    result_gene_summary_df['Level1_bootstrap_id'] = 'Size_'+ str(input_cas9_mouse_number) + '_Rep' + bootstrap_level1_cycle
    print('Level2&3 bootstrapping has finished')
    print('-------------------------------------------------------------------------------------------------')
    result_gRNA_summary_df.to_csv(output_address+'_'+bootstrap_level1_cycle+'_sgRNA.csv',index=False)
    result_gene_summary_df.to_csv(output_address+'_'+bootstrap_level1_cycle+'_gene.csv',index=False)

if __name__ == "__main__":
    main()  