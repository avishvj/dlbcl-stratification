import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr

# for imputation
import statsmodels.imputation.mice as mice
import statsmodels.regression.linear_model as lm

# this import statement will change depending on position of this .py file compared to utils.py
from src.utils import get_project_root
root = get_project_root()
data_path = root / 'data/reddy_data'

# note on openpxyl for windows?

class ReddyDataset():
    """Used to store data used for subsequent dimensionality reduction and analysis. 
    """
    # TODO: clinical_data have patient_id on index; sort out ordering of functions; sort out sample_id/patient_id naming
    def __init__(self):
        # should add one for all genes present
        self.patient_ids = None
        self.genetic_features = None
        self.expression_data = None
        self.norm_expression_data = None
        self.clinical_data = None
        self.subtypes = None

    def get_patient_subtype_annotations(self, clinical_data):
        # rename sample id to patient id, then make index, then create list
        c_data = clinical_data.copy()
        c_data.rename(columns={'Sample ID': 'PatientID', 'ABCGCB': 'COOClass'}, inplace=True)
        c_data.set_index('PatientID', inplace=True)
        # create subtype lists
        coo_list = c_data['COOClass']
        ipi_list = c_data['IPI']
        grm_list = c_data['GRM']
        patient_subtype_annotations = [coo_list, ipi_list, grm_list]
        dlbcl_subtypes = DLBCLSubtypes(patient_subtype_annotations)
        return dlbcl_subtypes
        
    def populate_all_attributes(self):
        # TODO: make order of these functions less important
        print("Reading in expression data...")
        reddy = self.create_expression_data()
        self.patient_ids = reddy.index
        reddy_norm = self.populate_norm_expression_data(reddy)
        print("Reading in clinical data...")
        reddy_clinical_initial = self.populate_clinical_data(reddy)
        print("\nImputing clinical data...")
        reddy_clinical_imputed = self.impute_clinical_data(reddy_clinical_initial)
        print()
        dlbcl_subtypes = self.get_patient_subtype_annotations(reddy_clinical_imputed)
        dlbcl_subtypes.populate_subtype_attributes()
        self.subtypes = dlbcl_subtypes
        reddy.rename(columns={'Sample ID': 'PatientID', 'ABCGCB': 'COOClass'}, inplace=True)
        reddy.set_index('PatientID', inplace=True) # set_index automatically removes feature column
        reddy_norm.rename(columns={'Sample ID': 'PatientID', 'ABCGCB': 'COOClass'}, inplace=True)
        reddy_norm.set_index('PatientID', inplace=True) # set_index automatically removes feature column
        self.expression_data = reddy
        self.norm_expression_data = reddy_norm
        print()
        final_message = """Successfully populated all variables. You can now access the ReddyDataset patient_ids, genetic_features, expression_data, norm_expression_data, clinical_data, and subtypes.
        """
        print(final_message)

    def create_expression_data(self):
        # transpose so columns: genes, rows: patients
        reddy_expression_path = data_path / 'reddy_tmm_quantile_log2_normalised_gene_count_matrix_RSEM.csv'
        reddy = np.transpose(pd.read_csv(reddy_expression_path, index_col=[0]))
        # remove null columns and store genetic features
        reddy = reddy.loc[:, reddy.columns.notnull()]
        self.genetic_features = reddy.columns
        # index is sample IDs, so add this as a column for merging on normal and z-score
        reddy['Sample ID'] = reddy.index
        reddy['Sample ID'] = reddy['Sample ID'].apply(pd.to_numeric)
        self.patient_ids = reddy.index
        # TODO: test patients = 775
        return reddy

    # change the name for this to create_norm_expression_data()
    def populate_norm_expression_data(self, expression_data):
        reddy_norm = expression_data.copy()
        if reddy_norm is None:
            reddy_norm = self.create_expression_data()
        # z-score on all columns excluding sample ID [takes a while]
        reddy_norm.loc[:, reddy_norm.columns != 'Sample ID'] = reddy_norm.loc[:, reddy_norm.columns != 'Sample ID'].apply(lambda x: x if np.std(x) == 0 else zscore(x))
        return reddy_norm

    def populate_clinical_data(self, reddy):
        reddy_info_prep = self.prep_clinical_data()
        # add gadd45b data by merging on sample ID
        reddy = reddy.iloc[:,:].copy()
        gadd45b_df = pd.DataFrame(reddy[['GADD45B', 'Sample ID']])
        reddy_info = pd.merge(reddy_info_prep, gadd45b_df, on='Sample ID', how='inner')
        # drop values with 5 or more NaNs
        reddy_info = reddy_info.dropna(thresh=6)
        # enumerate tumour purity feature ranges with middle of range
        col = reddy_info['Tumor Purity'].values
        for n, value in enumerate(col):
            if (value == '< 30%'):
                col[n] = 15
                continue
            if (value == '30 to 70%'):
                col[n] = 50
                continue
            if (value == '70% or more'):
                col[n] = 85
                continue
        # convert to float64 for regression
        reddy_info['Tumor Purity'] = reddy_info['Tumor Purity'].apply(pd.to_numeric)
        reddy_info['Tumor Purity'] = reddy_info['Tumor Purity'] * 0.01
        # changing 'response to initial therapy' to continuous values (adds randomness -> ideally would average)
        values = {'Response to initial therapy': 80}
        reddy_info = reddy_info.fillna(value=values)
        response_col = reddy_info['Response to initial therapy'].values
        for n, value in enumerate(response_col):
            if (value == 'No response'):
                response_col[n] = min(5, max(0, np.random.normal(2.5,1.5)))
                continue
            if (value == 'Partial response'):
                # maybe change to be on the lower side - SEE PG 490 REDDY
                response_col[n] = min(95, max(5, np.random.normal(55, 15)))
                continue
            if (value == 'Complete response'):
                response_col[n] = min(100, max(95, np.random.normal(95, 1.5)))
                continue
            if pd.isna(np.isnan(response_col[n])):
                print(n)
                response_col[n] = min(100, max(0, np.random.normal(60, 15)))
                continue
        # convert to float64 for regression
        reddy_info['Response to initial therapy'] = reddy_info['Response to initial therapy'].apply(pd.to_numeric)
        # rename columns
        reddy_info = reddy_info.rename(columns={"Response to initial therapy": 'ResponseToInitialTherapy'})
        reddy_info = reddy_info.rename(columns={"ABC GCB ratio (RNAseq)": 'Ratio'})
        reddy_info = reddy_info.rename(columns={"Genomic Risk Model": 'GRM'})
        reddy_info = reddy_info.rename(columns={"Overall survival years": 'OverallSurvivalYears'})
        reddy_info = reddy_info.rename(columns={"Tumor Purity": "TumorPurity"})
        reddy_info = reddy_info.rename(columns={"ABC GCB (RNAseq)": 'ABCGCB'})
        self.clinical_data = reddy_info
        return reddy_info

    def prep_clinical_data(self):
        """Helper function to prepare clinical data for merging with GADD45B in expression data.
        """
        # required subset of reddy_clinical_info
        reddy_s1_path = data_path / 'full_reddy_s1.xlsx'
        reddy_s1_clinical_info = pd.read_excel(reddy_s1_path, sheet_name='Clinical Information', engine='openpyxl')
        reddy_s1_clinical_info.columns = reddy_s1_clinical_info.iloc[2]
        reddy_s1_clinical_info = reddy_s1_clinical_info.drop([0,1,2])
        # select relevant info: don't need IPI groups feature since IPI more descriptive
        reddy_s1_clinical_info = reddy_s1_clinical_info[['Sample  ID', 'IPI', 'Response to initial therapy', 'Overall Survival years', 
        'Censored', 'ABC GCB (RNAseq)', 'ABC GCB ratio (RNAseq)', 'age at diagnosis', 'Genomic Risk Model']]
        reddy_s1_clinical_info = reddy_s1_clinical_info.rename(columns={"Sample  ID": "Sample ID", 
        "Overall Survival years": "Overall survival years", "age at diagnosis": "AgeAtDiagnosis"})
        # required subset of reddy sample level data
        reddy_s1_sample_level = pd.read_excel(reddy_s1_path, sheet_name='Sample Level Stats', engine='openpyxl')
        reddy_s1_sample_level.columns = reddy_s1_sample_level.iloc[1]
        reddy_s1_sample_level = reddy_s1_sample_level.drop([0,1])
        reddy_s1_sample_level = reddy_s1_sample_level[['sample_ID', 'Tumor Purity']]
        reddy_s1_sample_level = reddy_s1_sample_level.rename(columns={"sample_ID": 'Sample ID'})
        # merge two subsets on Sample ID
        reddy_info_prep = pd.merge(reddy_s1_clinical_info, reddy_s1_sample_level, on='Sample ID', how='inner')
        # enumerate GRM levels  - TODO: nan
        reddy_info_prep['Genomic Risk Model'] = reddy_info_prep['Genomic Risk Model'].replace({'Low risk': 1, 'Medium risk': 2, 'High risk': 3})
        # convert IPI, Overall survival years, ABC GCB ratio, Sample ID, AgeAtDiagnosis from object to int64/float64 for regression
        for feature_label in ['IPI', 'Overall survival years', 'ABC GCB ratio (RNAseq)', 'AgeAtDiagnosis', 'Sample ID']:
            reddy_info_prep[feature_label] = reddy_info_prep[feature_label].apply(pd.to_numeric) 
        return reddy_info_prep

    ##### START OF IMPUTATION FUNCTIONS

    # pass in reddy_info post initial population
    def impute_clinical_data(self, clinical_data):
        vars_to_impute = ['GRM', 'IPI', 'AgeAtDiagnosis', 'OverallSurvivalYears']
        clinical_data_imp = self.impute_reddy_info(vars_to_impute, clinical_data)
        return clinical_data_imp

    def impute_reddy_info(self, vars_to_impute, reddy_info):
        reddy_mice = reddy_info[['GRM', 'Ratio', 'GADD45B', 'AgeAtDiagnosis', 'ResponseToInitialTherapy', 'IPI', 'OverallSurvivalYears']]
        reddy_temp = reddy_info.copy()
        for var_name in vars_to_impute:
            reddy_temp = self.imputation_mice(var_name, reddy_mice, reddy_info)
        print("How many null values in 'Censored'? ", reddy_info['Censored'].isnull().sum())
        # 0 means censored, 1 means uncensored; assume null censors are 0
        values = {'Censored': 0}
        reddy_temp = reddy_temp.fillna(value=values)
        print("Censored values successfully imputed")
        if reddy_temp.isnull().values.any(): raise AssertionError ("There are still null values remaining.")
        else: print("We now have " + str(len(reddy_temp)) + " patients with complete data.")
        return reddy_temp

    def imputation_mice(self, var_name, reddy_mice, reddy_info):
        # not sure if this var_data is doing anything
        var_data = reddy_mice.copy()
        print("How many null values in " + var_name + " to change? " + str(reddy_info[var_name].isnull().sum()))
        imp_var = mice.MICEData(var_data)
        # create formula: var_name ~ sum(other_vars)
        other_vars = reddy_mice.loc[:, reddy_mice.columns != var_name].columns
        fml_var = var_name + ' ~' + ' +'.join(' {0}'.format(var) for var in other_vars)
        # perform mice imputation
        mice_var = mice.MICE(fml_var, lm.OLS, imp_var)
        results = mice_var.fit(10,10) # fit(#cycles to skip, #datasets to impute)
        reddy_info[var_name] = mice_var.data.data[var_name].values
        if reddy_info[var_name].isnull().sum() != 0:
            raise AssertionError ("All values could not be imputed.")
        else:
            #print(reddy_info[var_name].isnull().sum())
            print(var_name + " successfully imputed")
        return reddy_info

    ##### END OF IMPUTATION FUNCTIONS

class DLBCLSubtypes():
    """ Store the patient group labels for each of potential subtypes e.g. subtype is COOClass -> patients will have one of ABC, GCB, Unclassified group label.
        TODO: add some sort of distribution creation functions that can help with the subtype extension framework.
        TODO: clarify type of each args in the functions.
        TODO: generalise create_subtype_colour_info() function for superclass i.e. unknown subtypes that can all be diff colours.
    """
    def __init__(self, all_patient_subtype_annotations: list):
        self.all_patient_subtype_annotations = all_patient_subtype_annotations
        self.subtype_colour_info_list = None
    
    # main func to run
    def populate_subtype_attributes(self):
        print("Populating subtype attributes...")
        subtype_colour_info_list = self.create_all_subtypes_colour_info(self.all_patient_subtype_annotations)
        self.subtype_colour_info_list = subtype_colour_info_list
        print("All subtype attributes populated - can now be used for visualisation.")
    
    # subtype colour info list = [(s1_cm, s1_b), (s2_cm, s2_b)]
    def create_all_subtypes_colour_info(self, all_patient_subtype_annotations):
        subtype_colour_info_list = []
        for specific_subtype_annotations in all_patient_subtype_annotations:
            subtype_colour_info = self.create_subtype_colour_info(specific_subtype_annotations)
            subtype_colour_info_list.append(subtype_colour_info)
        return subtype_colour_info_list 

    def create_subtype_colour_info(self, specific_subtype_annotations):
        subtype_bar = None
        subtype_colourmap = None
        if specific_subtype_annotations is None:
            raise TypeError("Subtype is not defined yet.")
        if specific_subtype_annotations.name == 'COOClass':
            subtype_bar = dict(zip(specific_subtype_annotations.unique(), "rgb"))
            subtype_colourmap = specific_subtype_annotations.map(subtype_bar)
            subtype_colourmap.rename("COO Class", inplace=True)
        elif specific_subtype_annotations.name == 'GRM':
            subtype_bar = dict(zip(specific_subtype_annotations.unique(), "cmy"))
            subtype_colourmap = specific_subtype_annotations.map(subtype_bar)
            subtype_colourmap.rename("GRM Level", inplace=True)
            subtype_bar = dict(sorted(subtype_bar.items(), key=lambda x:x[0])) # sort so smallest value first
        # TODO: add IPI
        return subtype_colourmap, subtype_bar