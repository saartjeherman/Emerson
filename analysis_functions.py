import gzip
import pandas as pd
import os
from collections import defaultdict
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import ast
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from joblib import dump, load

import matplotlib.pyplot as plt
import seaborn as sns

from raptcr.io.pipeline import ProcessingPipeline
from raptcr.io.mappers import RegexMapper
from raptcr.neighborhood import ConvergenceAnalysis, Fisher
from raptcr.hashing import TCRDistEmbedder
from clustcrdist.background import BackgroundModel

path = os.getcwd()


"""
Match patients with TCRs
# Read all the patient files and make a 
# dictionary where key = combination, value = list of reperoire_ids
"""

def read_patient_files(patients_path, number_of_patients=666, training=True):
    """
    Reads and processes patient file names from a specified directory.
    Args:
        patients_path (str): The relative path to the directory containing patient files.
        number_of_patients (int, optional): The number of patient files to include in the training set. 
            Defaults to 666.
        training (bool, optional): A flag indicating whether to process files for training or testing. 
            If True, selects the first `number_of_patients` files. If False, selects files after the 
            first `number_of_patients`. Defaults to True.
    Returns:
        tuple: A tuple containing:
            - tsv_gz_files (list of str): A list of file names with the '.tsv.gz' extension.
            - patient_names (list of str): A list of patient names extracted from the file names 
              (file names without extensions).
    """
    files_path = path + patients_path #'\\data\\HLA_emerson_2017'

    all_files = os.listdir(files_path)

    tsv_gz_files = [f for f in all_files if f.endswith('.tsv.gz')]

    if training == True:
        tsv_gz_files = tsv_gz_files[:number_of_patients] #tsv_gz_files[:400]

    elif training == False:
        tsv_gz_files = tsv_gz_files[number_of_patients:]

    patient_names = [f.split('.')[0] for f in tsv_gz_files]

    return tsv_gz_files, patient_names



def matching_patients_tcrs(patients_path, number_of_patients=400, top_n=None, min_occurences=5):
    """
    Analyzes T-cell receptor (TCR) data from patient files and identifies clonotypes 
    (unique combinations of TCR features) that occur in multiple patients.
    Args:
        patients_path (str): Path to the directory containing patient data files.
        number_of_patients (int, optional): Maximum number of patient files to process. 
            Defaults to 400.
        top_n (int, optional): If specified, only the top N rows of each patient file 
            are considered. Defaults to None.
        min_occurences (int, optional): Minimum number of patients in which a clonotype 
            must appear to be included in the output. Defaults to 5.
    Returns:
        pd.DataFrame: A DataFrame containing clonotypes and the corresponding repertoire IDs 
        of patients in which they occur. The DataFrame has the following columns:
            - 'combination': A tuple representing the clonotype (v_call, junction_aa, j_call).
            - 'repertoire_ids': A set of repertoire IDs where the clonotype is found.
    """
    tsv_gz_files, patient_names = read_patient_files(patients_path, number_of_patients)

    files_path = path + patients_path #'\\data\\HLA_emerson_2017'

    clonotype_occurences = defaultdict(set)

    for f in tsv_gz_files:
        file_path = os.path.join(files_path, f)
        with gzip.open(file_path, 'rt') as fi:
            # Inlezen als pandas DataFrame
            df = pd.read_csv(fi, sep='\t')
            # Select only top_n rows if top_n is not None and top_n is less than the number of rows
            if top_n is not None and top_n < len(df):
                df = df.head(top_n)

            for row in df.itertuples(index=False):
                combination = (row.v_call, row.junction_aa, row.j_call)
                rep_id = row.repertoire_id
                clonotype_occurences[combination].add(rep_id)


    tcr_df = pd.DataFrame(list(clonotype_occurences.items()), columns=['combination', 'repertoire_ids'])
    tcr_df = tcr_df[tcr_df['repertoire_ids'].apply(len) >= min_occurences]
    tcr_df.reset_index(drop=True, inplace=True)
    return tcr_df



"""
Read and write to files functions
"""

def write_to_file(data, filename):
    data.to_csv(filename, sep='\t', index=False)


def read_file(filename):
    return pd.read_csv(filename, sep='\t')



"""
2. Match HLA-labelling of patients
# Match the HLA-labelling with the patient files 
# (A_02_01_features.txt contains the indices to the patient files)
"""

def match_hla(patients_path, hla_path, hla_type='HLA-A*02:01'):
    """
    Matches patients to a specific HLA type and returns a DataFrame indicating the presence or absence of the HLA type.
    Args:
        patients_path (str): Path to the directory containing patient files.
        hla_path (str): Path to the file containing HLA data.
        hla_type (str, optional): The HLA type to match (default is 'HLA-A*02:01').
    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - 'repertoire_id': The patient identifier.
            - 'has_<hla_type>': Boolean indicating if the patient has the specified HLA type (True, False, or None if not determined).
    Notes:
        - The function assumes that `read_patient_files` is defined elsewhere and returns a list of file paths and patient names.
        - The HLA data file is expected to have a specific format where the HLA type is in the second column, and patient indices are listed starting from the sixth column.
        - The column name in the resulting DataFrame is dynamically generated based on the `hla_type` parameter, with special characters replaced for compatibility.
    """
    tsv_gz_files, all_patient_names = read_patient_files(patients_path)

    record =  {}  # Temporary dictionary to hold one record
    file_path = path + hla_path #'\\data\\A_02_01_features.txt'

    # Read and parse the file
    with open(file_path, "r") as file:
        for line in file:
            line = line.split(" ")
            if line[1] == hla_type:
                key = line[2]
                for i in range(5, len(line)):
                    value = line[i]
                    if value == '\n':
                        continue
                    ## value is the index to the patient file
                    patient = all_patient_names[int(value)]
                    if key in ['num_positives:']:
                        record[patient] = True
                    elif key in ['num_negatives:']:
                        record[patient] = False   


    for patient in all_patient_names:
        if patient not in record.keys():
            record[patient] = None

    column_name = 'has_' + hla_type.replace('*', '').replace(':', '_').replace('-', '_')

    patient_df = pd.DataFrame(list(record.items()), columns=['repertoire_id', column_name])
    patient_df.sort_values(by=['repertoire_id'], inplace=True, ignore_index=True)
    return patient_df


## select number of patients of the hla labelling for training or testing and remove rows with missing values
def select_patients_hla_labelling(patient_df, number_of_patients=666, hla_column='has_HLA_A02_01', training=True):
    """
    Filters and selects a subset of patients based on HLA labelling and training mode.

    Args:
        patient_df (pd.DataFrame): The DataFrame containing patient data.
        number_of_patients (int, optional): The number of patients to include in the training or testing set. 
            Defaults to 666.
        hla_column (str, optional): The column name indicating HLA labelling. Defaults to 'has_HLA_A02_01'.
        training (bool, optional): If True, selects the first `number_of_patients` rows for training. 
            If False, selects rows after `number_of_patients` for testing. Defaults to True.

    Returns:
        pd.DataFrame: A filtered and reset DataFrame containing the selected patients with non-null HLA labelling.
    """
    if training == True:
        patient_df = patient_df.iloc[:number_of_patients]
    elif training == False:
        patient_df = patient_df.iloc[number_of_patients:]
    patient_df = patient_df[patient_df[hla_column].notna()]
    patient_df.reset_index(drop=True, inplace=True)    
    return patient_df



"""
3. Fisher-exact method
# Make a dataframe where every row contains the TCR (combination), 
# the p-value and odds ratio of fisher_exact method (use psuedocount of 0.1)
"""

def execute_fisher_exact(tcr_df, patient_df, hla_column='has_HLA_A02_01', hla_type='HLA-A*02:01'):
    """
    Performs Fisher's Exact Test on TCR (T-cell receptor) and HLA (human leukocyte antigen) data.
    This function calculates the association between specific TCR combinations and the presence of a specific HLA type
    using Fisher's Exact Test. It generates a 2x2 contingency table for each TCR combination and computes the odds ratio
    and p-value. The results are returned in a DataFrame with additional statistical metrics.
    Args:
        tcr_df (pd.DataFrame): A DataFrame containing TCR data. It must include the columns:
            - 'repertoire_ids': A list of repertoire IDs associated with each TCR combination.
            - 'combination': The TCR combination being analyzed.
        patient_df (pd.DataFrame): A DataFrame containing patient data. It must include the columns:
            - 'repertoire_id': The repertoire ID for each patient.
            - hla_column (str): A boolean column indicating whether the patient has the specified HLA type.
        hla_column (str, optional): The column name in `patient_df` indicating the presence of the HLA type. Defaults to 'has_HLA_A02_01'.
        hla_type (str, optional): The HLA type being analyzed. Defaults to 'HLA-A*02:01'.
    Returns:
        pd.DataFrame: A DataFrame containing the results of Fisher's Exact Test for each TCR combination. The columns include:
            - 'HLA': The HLA type being analyzed.
            - 'TCR': The TCR combination being analyzed.
            - 'odds_ratio': The odds ratio from Fisher's Exact Test.
            - 'p_value': The p-value from Fisher's Exact Test.
            - 'have_hla_and_tcr': Count of patients with both the HLA type and the TCR combination.
            - 'have_no_hla_and_tcr': Count of patients without the HLA type but with the TCR combination.
            - 'have_hla_no_tcr': Count of patients with the HLA type but without the TCR combination.
            - 'have_no_hla_no_tcr': Count of patients without both the HLA type and the TCR combination.
            - 'log2_odds_ratio': The log2-transformed odds ratio.
            - 'neg_log10_p_value': The negative log10-transformed p-value.
            - 'significant': A boolean indicating whether the p-value is less than 0.05.
    """
    fisher_exact_results = []
    for index, row in tcr_df.iterrows():
        # Haal de eerste rij van tcr_df en zijn combinatie en repertoire_ids
        repertoire_ids = row['repertoire_ids']  # Aangenomen dat repertoire_ids een lijst is
        combination = row['combination']

        # Filter patient_df op basis van de repertoire_ids
        filtered_patients = patient_df[patient_df['repertoire_id'].isin(repertoire_ids)]

        # Tel het aantal patiënten in elke categorie
        have_hla_and_tcr = len(filtered_patients[filtered_patients[hla_column] == True]) + 0.1  # Aantal met HLA en TCR
        have_no_hla_and_tcr = len(filtered_patients[filtered_patients[hla_column] == False]) + 0.1  # Aantal zonder HLA maar met TCR


        # Aantal patiënten zonder TCR
        have_hla_no_tcr = len(patient_df[(patient_df[hla_column] == True) & 
                                                (~patient_df['repertoire_id'].isin(repertoire_ids))]) + 0.1
        have_no_hla_no_tcr = len(patient_df[(patient_df[hla_column] == False) & 
                                                    (~patient_df['repertoire_id'].isin(repertoire_ids))]) + 0.1
        

        # Maak de 2x2-contingentietabel
        contingency_table = [
            [
                have_hla_and_tcr,  # a: patiënten met zowel HLA als TCR
                have_no_hla_and_tcr  # b: patiënten zonder HLA maar met TCR
            ],
            [
                have_hla_no_tcr,  # c: patiënten met HLA maar zonder TCR
                have_no_hla_no_tcr  # d: patiënten zonder HLA en zonder TCR
            ]
        ]


        # Voer de Fisher's Exact Test uit
        odds_ratio, p_value = fisher_exact(contingency_table)


        # Voeg de resultaten toe aan de lijst
        fisher_exact_results.append({
            'HLA': hla_type,
            'TCR': combination,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'have_hla_and_tcr' : have_hla_and_tcr,
            'have_no_hla_and_tcr': have_no_hla_and_tcr,
            'have_hla_no_tcr': have_hla_no_tcr,
            'have_no_hla_no_tcr': have_no_hla_no_tcr
        })

    # Zet de resultaten om in een DataFrame
    fisher_exact_results_df = pd.DataFrame(fisher_exact_results)

    # Neem de -log10 van de p-value en de log2 van de odds_ratio en voeg ze toe aan de DataFrame
    fisher_exact_results_df['log2_odds_ratio'] = np.log2(fisher_exact_results_df['odds_ratio'])
    fisher_exact_results_df['neg_log10_p_value'] = -np.log10(fisher_exact_results_df['p_value'])

    # Voeg een kolom toe om te controleren of de p-value kleiner is dan 0.05
    fisher_exact_results_df['significant'] = fisher_exact_results_df['p_value'] < 0.05
    return fisher_exact_results_df



def generate_volcano_plot(fisher_exact_results_df, filename):
    # Maak een Volcano Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=fisher_exact_results_df, 
                    x='log2_odds_ratio', 
                    y='neg_log10_p_value',
                    hue='significant')
    plt.axhline(y=-np.log10(0.05), color='grey', linestyle='--')
    plt.xlabel('log2(Odds Ratio)')
    plt.ylabel('-log10(p-value)')
    plt.title('Volcano Plot')
    plt.savefig(filename)
    plt.show()



"""
4. Train classifier model
# match/count the HLA-02.01 related TCR's for every patient
# make a table with 3 columns: 
# patient, #total TCR's and #HLA-02.01 related TCR's 
# (from the fisher_exact_results_df where p-value < 0.05 and odds_ratio > 1)
"""

def apply_benjamini_hochberg(fisher_exact_results_df):
    # Voer de Benjamini-Hochberg-correctie uit
    fisher_exact_results_df = fisher_exact_results_df.sort_values(by='p_value').reset_index(drop=True)
    fisher_exact_results_df['corrected_p_value'] = fisher_exact_results_df['p_value'] * len(fisher_exact_results_df) / (fisher_exact_results_df.index + 1)

    return fisher_exact_results_df



def match_and_select_related_tcrs(fisher_exact_results_df, hla_labelling, patients_path, 
                                  top_n=10, number_patients=666, sort_column='p_value', 
                                  benjamini_hochberg=False, selected_tcrs_patients=None, hla_column='has_HLA_A02_01',
                                  feature_importance=False):
    """
    Matches and selects related T-cell receptors (TCRs) based on statistical results and patient data.
    Args:
        fisher_exact_results_df (pd.DataFrame): DataFrame containing Fisher's exact test results with columns 
            such as 'p_value', 'odds_ratio', and optionally 'coefficient'.
        hla_labelling (pd.DataFrame): DataFrame containing HLA labelling information for patients, including 
            'repertoire_id' and HLA-related columns.
        patients_path (str): Path to the directory containing patient data files.
        top_n (int, optional): Number of top TCRs to select. Defaults to 10.
        number_patients (int, optional): Number of patient files to process. Defaults to 666.
        sort_column (str, optional): Column to sort the TCRs by ('p_value' or 'odds_ratio'). Defaults to 'p_value'.
        benjamini_hochberg (bool, optional): Whether to apply the Benjamini-Hochberg correction for multiple testing. 
            Defaults to False.
        selected_tcrs_patients (int, optional): Number of TCRs to select per patient. If None, all TCRs are used. 
            Defaults to None.
        hla_column (str, optional): Column name in `hla_labelling` indicating the presence of a specific HLA type. 
            Defaults to 'has_HLA_A02_01'.
        feature_importance (bool, optional): Whether to select TCRs based on feature importance (using 'coefficient' 
            column). Defaults to False.
    Returns:
        pd.DataFrame: DataFrame containing the results for each patient, including:
            - 'repertoire_id': Patient repertoire ID.
            - 'total_tcrs': Total number of TCRs in the patient data.
            - 'related_tcrs': Number of TCRs matching the selected top TCRs.
            - HLA column (e.g., 'has_HLA_A02_01'): Presence of the specified HLA type for the patient.
        int: Number of top TCRs selected if fewer than `top_n`.
    Notes:
        - The function processes patient data files in `.tsv.gz` format.
        - TCRs are identified as tuples of ('v_call', 'junction_aa', 'j_call').
        - If `feature_importance` is True, TCRs are selected based on the 'coefficient' column in descending order.
        - If `benjamini_hochberg` is True, the Benjamini-Hochberg correction is applied to p-values, and TCRs are 
            selected based on the corrected p-values.
        - The function handles missing or NaN values in the HLA column by setting them to None.
    """
                                  
    

    tsv_gz_files, patient_names = read_patient_files(patients_path, number_patients)

    files_path = path + patients_path #'\\data\\HLA_emerson_2017'

    result_data = []
    top_related_tcrs = None

    if feature_importance:
        # Sort the fisher_exact_results_df by coefficient in descending order
        fisher_exact_results_df = fisher_exact_results_df.sort_values(by=['coefficient'], ascending=False).reset_index(drop=True)
        # Select the top_n TCRs based on coefficient
        top_related_tcrs = fisher_exact_results_df.head(top_n)
        
    else:
        related_tcrs = fisher_exact_results_df[(fisher_exact_results_df['p_value'] < 0.05) & (fisher_exact_results_df['odds_ratio'] > 1)].reset_index(drop=True)
        #print("Related TCRs fisher exact: ",len(related_tcrs)) 
        # if sort column is p_value, sort the related tcrs by p_value in ascending order (lower p-value is better)
        top_related_tcrs = None
        if sort_column == 'p_value':
            top_related_tcrs = related_tcrs.sort_values(by=['p_value']).head(top_n)
        # if sort column is odds_ratio, sort the related tcrs by odds_ratio in descending order (higher odds ratio is better)
        elif sort_column == 'odds_ratio':
            related_tcrs = related_tcrs.sort_values(by=['odds_ratio'], ascending=False)
            top_related_tcrs = related_tcrs.head(top_n)

        if benjamini_hochberg:
            fisher_exact_results_df = apply_benjamini_hochberg(fisher_exact_results_df)
            related_tcrs = fisher_exact_results_df[(fisher_exact_results_df['corrected_p_value'] < 0.05) & (fisher_exact_results_df['odds_ratio'] > 1)].reset_index(drop=True)
            if sort_column == 'p_value':
                top_related_tcrs = related_tcrs.sort_values(by=['corrected_p_value']).head(top_n)
            else:
                top_related_tcrs = related_tcrs.sort_values(by=[sort_column]).head(top_n)

    print("Selected related TCRs: ",len(top_related_tcrs)) 
    #print("Selected related TCRs: ",top_related_tcrs)   

    #check if top-related tcrs has column 'TCR'
    if 'TCR' not in top_related_tcrs.columns:
        top_related_tcrs['TCR'] = [tuple(tcr) for tcr in zip(top_related_tcrs['v_call'], top_related_tcrs['junction_aa'], top_related_tcrs['j_call'])]

    
    for f in tsv_gz_files:
        file_path = os.path.join(files_path, f)
        with gzip.open(file_path, 'rt') as fi:
            # Inlezen als pandas DataFrame
            df = pd.read_csv(fi, sep='\t')
            
            # Controleer of DataFrame niet leeg is
            if df.empty:
                continue

            if selected_tcrs_patients is not None:
                df = df.head(selected_tcrs_patients)
                #print("Selected TCRs per patient: ",selected_tcrs_patients)
            
            # Voeg de TCR-combinaties toe als tuple
            if {'v_call', 'junction_aa', 'j_call'}.issubset(df.columns):
                df['TCR'] = [tuple(tcr) for tcr in zip(df['v_call'], df['junction_aa'], df['j_call'])]            
            else:
                #print("oclumns are not present")
                continue  # Sla over als de vereiste kolommen ontbreken
            
            # Haal totalen op
            total_tcrs = len(df)
            rep_id = df['repertoire_id'].iloc[0]  # Gebruik .iloc om een fout te voorkomen

            # check if value of HLA-A*02:01 is NaN
            has_hla = None
            hla_row = hla_labelling[hla_labelling['repertoire_id'] == rep_id]
            if not hla_row.empty:
                has_hla = hla_row[hla_column].iloc[0]
            #if hla is NaN value, change to None
            if pd.isna(has_hla):  # Controleer of de waarde NaN is
                has_hla = None
            
            # Zoek naar matches met de significante TCRs
            matching_tcrs = df[df['TCR'].isin(top_related_tcrs['TCR'])].shape[0]
            
            # Voeg de resultaten toe aan de lijst
            result_data.append({
                'repertoire_id': rep_id,
                'total_tcrs': total_tcrs,
                'related_tcrs': matching_tcrs,
                hla_column: has_hla
            })
            

    # Als je de data als een DataFrame wilt hebben
    result_df = pd.DataFrame(result_data)

    if len(top_related_tcrs) < top_n:
        return result_df, len(top_related_tcrs)

    return result_df 

    
def calculate_metrics(result_df, hla_column='has_HLA_A02_01'):
    """
    Calculate and display performance metrics for a classification model.
    This function computes the confusion matrix and calculates sensitivity 
    (True Positive Rate), specificity (True Negative Rate), and overall accuracy 
    for a given classification result. It also prints a classification report.
    Args:
        result_df (pd.DataFrame): A pandas DataFrame containing the classification 
            results. It must include columns for the true labels and predictions.
        hla_column (str, optional): The base name of the column representing the 
            HLA type. The function expects the true label column to be named 
            '<hla_column>_label'. Defaults to 'has_HLA_A02_01'.
    Returns:
        tuple: A tuple containing the following metrics:
            - sensitivity (float): The True Positive Rate.
            - specificity (float): The True Negative Rate.
            - accuracy (float): The overall accuracy of the model.
    Prints:
        - Sensitivity, specificity, and accuracy values.
        - A classification report with precision, recall, and F1-score for each class.
    Notes:
        - The function assumes that the true labels are binary (0 or 1).
        - The `confusion_matrix` and `classification_report` functions from 
          `sklearn.metrics` are used for calculations.
    """
    # Compute the confusion matrix
    hla_label = hla_column +'_label'
    cm = confusion_matrix(result_df[hla_label], result_df['prediction'])
    TN, FP, FN, TP = cm.ravel()

    # Calculate metrics
    sensitivity = TP / (TP + FN)  # True Positive Rate
    specificity = TN / (TN + FP)  # True Negative Rate
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # Overall accuracy

    # Print results
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    print(classification_report(result_df[hla_label], result_df['prediction'], target_names=['Class 0', 'Class 1']))
    return sensitivity, specificity, accuracy


#make a ROC curve for the trained model
def plot_roc_curve(result_df, top_n, training=True, benjamini_hochberg=False, 
                   threshold=None, selected_tcrs_patients=None, semi_supervised=False, 
                   convergence_method=False, folder='results\\', sort_column='p_value', hla_column='has_HLA_A02_01'):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and calculates the Area Under the Curve (AUC) 
    for a logistic regression model based on the given results dataframe.
    Args:
        result_df (pd.DataFrame): DataFrame containing the results, including columns for true labels 
                                    and predicted probabilities.
        top_n (int): The number of top related TCRs to consider.
        training (bool, optional): Whether the data is from the training set. Defaults to True.
        benjamini_hochberg (bool, optional): Whether to apply the Benjamini-Hochberg correction. Defaults to False.
        threshold (float, optional): Threshold value for filtering. Defaults to None.
        selected_tcrs_patients (str, optional): Identifier for selected TCRs patients. Defaults to None.
        semi_supervised (bool, optional): Whether the model is semi-supervised. Defaults to False.
        convergence_method (bool, optional): Whether a convergence method was used. Defaults to False.
        folder (str, optional): Path to the folder where the plot will be saved. Defaults to 'results\\'.
        sort_column (str, optional): Column used for sorting ('p_value' or 'odds_ratio'). Defaults to 'p_value'.
        hla_column (str, optional): Column name indicating the HLA type. Defaults to 'has_HLA_A02_01'.
    Returns:
        float: The calculated AUC value.
    Notes:
        - The function generates a plot of the ROC curve and saves it as a PNG file in the specified folder.
        - The plot title and filename are dynamically adjusted based on the provided arguments.
        - The function assumes that `result_df` contains columns for true labels (e.g., `hla_column + '_label'`) 
            and predicted probabilities ('probability').
    """
                   
    # Bereken ROC-curve en AUC
    hla_label = hla_column +'_label'
    hla_type = hla_column.replace('has_', '')
    fpr, tpr, thresholds = roc_curve(result_df[hla_label], result_df['probability'])
    roc_auc = auc(fpr, tpr)

    # Plot instellingen
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Bepaal titel en bestandsnaam
    dataset_type = "training" if training else "validation"
    title = f"ROC Logistic Regression {hla_type} top {top_n} related tcrs ({dataset_type.capitalize()} set)"
    filename = folder + f"\\plots\\ROC_Logistic_Regression_{hla_type}_top{top_n}_related_tcrs_{dataset_type}.png"
    #filename = f"results\\plots\\ROC_Logistic_Regression_top{top_n}_related_tcrs_{dataset_type}.png"

    if benjamini_hochberg:
        title += ", Benjamini-Hochberg"
        filename = filename.replace(".png", "_bh.png")
    if threshold is not None:
        title += f", Threshold = {threshold}"
        filename = filename.replace(".png", f"_threshold_{threshold}.png")
    if selected_tcrs_patients is not None:
        title += f", Selected TCRs Patients = {selected_tcrs_patients}"
        filename = filename.replace(".png", f"_selected_tcrs_patients_{selected_tcrs_patients}.png")
    if semi_supervised:
        title += ", Semi-supervised"
        filename = filename.replace(".png", "_semi_supervised.png")
    if convergence_method:
        title += ", Convergence method"
        filename = filename.replace(".png", "_convergence_method.png")
    if sort_column == 'p_value':
        title += ", Sorted by p-value"
        filename = filename.replace(".png", "_sorted_p_value.png")
    elif sort_column == 'odds_ratio':
        title += ", Sorted by odds ratio"
        filename = filename.replace(".png", "_sorted_odds_ratio.png")

    # Toon en sla de plot op
    print(title)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.5)
    plt.savefig(filename)
    plt.show()

    return roc_auc



def make_classifier(result_df, top_n=10, benjamini_hochberg=False, filename=None, hla_column='has_HLA_A02_01', weights=None):
    """
    Creates and trains a logistic regression classifier using the provided DataFrame.
    Args:
        result_df (pd.DataFrame): A DataFrame containing the features and target variable.
                                  Expected columns include 'total_tcrs', 'related_tcrs', and a column
                                  specified by `hla_column` suffixed with '_label' for the target.
        top_n (int, optional): The number of top related TCRs to consider. Defaults to 10.
        benjamini_hochberg (bool, optional): If True, applies the Benjamini-Hochberg correction and 
                                             saves the model with a specific filename. Defaults to False.
        filename (str, optional): The path to save the trained model. If None, a default filename is used. Defaults to None.
        hla_column (str, optional): The column name representing the HLA type. Defaults to 'has_HLA_A02_01'.
        weights (str or bool, optional): Class weights for the logistic regression model. If True, uses 'balanced'. Defaults to None.
    Returns:
        tuple: A tuple containing:
            - result_df (pd.DataFrame): The input DataFrame with added columns for predicted probabilities 
                                        ('probability') and predictions ('prediction').
            - model (LogisticRegression): The trained logistic regression model.
    Notes:
        - The model is trained using 'total_tcrs' and 'related_tcrs' as features.
        - The target variable is derived from the `hla_column` suffixed with '_label'.
        - The model is saved to a file, with the filename determined by `benjamini_hochberg` and `filename`.
    """
    # make classifier logistic regression: 
    # - features are total_tcrs and related_tcrs
    # - target is has_HLA_A02_01 
    # - save the model

    if weights is True:
        weights = 'balanced'
    model = LogisticRegression(class_weight=weights)
    hla_label = hla_column +'_label'
    X = result_df[['total_tcrs', 'related_tcrs']]
    y = result_df[hla_label]
    model.fit(X, y)
    
    result_df['probability'] = model.predict_proba(X)[:, 1]
    result_df['prediction'] = model.predict(X)


    #save the model
    if benjamini_hochberg and filename == None:
        dump(model, 'results\\models\\logistic_regression_top' + str(top_n) + '_related_tcrs_bh.joblib')
    elif benjamini_hochberg == False and filename == None:
        dump(model, 'results\\models\\logistic_regression_top' + str(top_n) + '_related_tcrs.joblib')

    else:
        dump(model, filename)

    return result_df, model



def plot_total_vs_significant_tcrs(result_df, top_n=10, training=True, threshold=None):
    # Zorg ervoor dat de kolom consistente typen heeft
    # Zet None om naar een string 'None'
    result_df['has_HLA_A02_01'] = result_df['has_HLA_A02_01'].map(
        {True: True, False: False, None: 'None'}
    )

    plt.figure(figsize=(8, 6))  # Stel de grootte van de plot in
    palette = {True: 'green', False: 'red', 'None': 'blue'}

    # Maak de scatterplot met juiste kleurtoewijzing
    sns.scatterplot(
        x=result_df['total_tcrs'], 
        y=result_df['related_tcrs'], 
        hue=result_df['has_HLA_A02_01'],  # Gebruik de aangepaste kolom voor 'hue'
        palette=palette  # Geef kleuren op via het palette
    )

    # Voeg labels en titel toe
    plt.xlabel('# Total TCRs')
    plt.ylabel('# Significant TCRs')
    plt.legend(title='Has HLA-A*02:01', loc='upper right')
    
    if training == True:
        if threshold != None:
            plt.title('Total vs Significant TCRs per Patient TCRs(Training set, Top ' + str(top_n) + ' TCRs, Threshold ' + str(threshold) + ')')
            plt.savefig('results\\plots\\Total vs Significant TCRs per Patient_TCRs(Training set, Top ' + str(top_n) + ' TCRs, Threshold ' + str(threshold) + ').png')
        else:
            plt.title('Total vs Significant TCRs per Patient TCRs(Training set, Top ' + str(top_n) + ' TCRs)')
            plt.savefig('results\\plots\\Total vs Significant TCRs per Patient_TCRs(Training set, Top ' + str(top_n) + ' TCRs)')
    else:
        if threshold != None:
            plt.title('Total vs Significant TCRs per Patient TCRs(Validation set, Top ' + str(top_n) + ' TCRs, Threshold ' + str(threshold) + ')')
            plt.savefig('results\\plots\\Total vs Significant TCRs per Patient_TCRs(Validation set, Top ' + str(top_n) + ' TCRs, Threshold ' + str(threshold) + ').png')
        else:
            plt.title('Total vs Significant TCRs per Patient TCRs(Validation set, Top ' + str(top_n) + ' TCRs)')
            plt.savefig('results\\plots\\Total vs Significant TCRs per Patient_TCRs(Validation set, Top ' + str(top_n) + ' TCRs)')
            
    # Toon de plot
    plt.show()



def preprocessing_dataset(fisher_exact_results_df, patient_df, patients_path, 
                          top_n=10, benjamini_hochberg=False, selected_tcrs_patients=None, sort_column='p_value',
                          hla_column='has_HLA_A02_01', feature_importance=False):
    

    """
    Preprocesses the dataset by matching and selecting related TCRs, handling missing values, 
    and encoding the target column for modeling.
    Args:
        fisher_exact_results_df (pd.DataFrame): DataFrame containing Fisher's exact test results.
        patient_df (pd.DataFrame): DataFrame containing patient information.
        patients_path (str): Path to the patient data file.
        top_n (int, optional): Number of top TCRs to select. Defaults to 10.
        benjamini_hochberg (bool, optional): Whether to apply the Benjamini-Hochberg correction. Defaults to False.
        selected_tcrs_patients (list, optional): List of selected TCRs for specific patients. Defaults to None.
        sort_column (str, optional): Column name to sort the results by. Defaults to 'p_value'.
        hla_column (str, optional): Column name for the HLA label. Defaults to 'has_HLA_A02_01'.
        feature_importance (bool, optional): Whether to include feature importance in the processing. Defaults to False.
    Returns:
        pd.DataFrame: Processed DataFrame with selected TCRs, encoded target labels, and no missing values.
    """

    result_df = match_and_select_related_tcrs(fisher_exact_results_df=fisher_exact_results_df, hla_labelling=patient_df, 
                                              patients_path=patients_path, top_n=top_n, sort_column=sort_column,
                                              benjamini_hochberg=benjamini_hochberg, selected_tcrs_patients=selected_tcrs_patients, 
                                              hla_column=hla_column, feature_importance=feature_importance)
    
    # Make a label for the target (has_HLA_A02_01), model can't handle boolean values
    # 0 = False, 1 = True
    result_df[hla_column] = result_df[hla_column].replace('None', np.nan)
    #drop none values
    result_df = result_df.dropna()
    hla_label = hla_column +'_label'
    result_df[hla_label] = result_df[hla_column].astype('category').cat.codes
    #print("result_df: ",result_df)
    return result_df


def train_training_dataset(result_df_training, top_n=10, benjamini_hochberg=False, 
                            threshold=None, selected_tcrs_patients=None, filename_model=None,
                           semi_supervised=False, convergence_method=False, folder='results\\',
                           sort_column='p_value', hla_column='has_HLA_A02_01', weights=None):
    """
    Trains a classifier on a training dataset and evaluates its performance.
    Parameters:
        result_df_training (pd.DataFrame): The input DataFrame containing training data.
        top_n (int, optional): The number of top features to consider for training. Default is 10.
        benjamini_hochberg (bool, optional): Whether to apply the Benjamini-Hochberg procedure. Default is False.
        threshold (float, optional): Threshold value for feature selection. Default is None.
        selected_tcrs_patients (list, optional): List of selected TCRs for specific patients. Default is None.
        filename_model (str, optional): Path to save the trained model. Default is None.
        semi_supervised (bool, optional): Whether to use semi-supervised learning. Default is False.
        convergence_method (bool, optional): Whether to use a convergence method during training. Default is False.
        folder (str, optional): Directory to save results. Default is 'results\\'.
        sort_column (str, optional): Column name to sort the data by. Default is 'p_value'.
        hla_column (str, optional): Column name indicating HLA information. Default is 'has_HLA_A02_01'.
        weights (dict, optional): Weights for the classifier. Default is None.
    Returns:
        model: The trained classifier model.
        metrics_training_data (dict): A dictionary containing training metrics, including:
            - 'top_n': The number of top features used.
            - 'threshold': The threshold value used.
            - 'selected_tcrs_patients': The selected TCRs for specific patients.
            - 'benjamini_hochberg': Whether the Benjamini-Hochberg procedure was applied.
            - 'sort_column': The column used for sorting.
            - 'roc_auc': The ROC AUC score.
            - 'sensitivity': The sensitivity of the model.
            - 'specificity': The specificity of the model.
            - 'accuracy': The accuracy of the model.
    """
                          
    
    #select first 400 patients for training
    result_df_training.sort_values(by=['repertoire_id'], inplace=True, ignore_index=True)
    if semi_supervised == False:    
        result_df_training = result_df_training.iloc[:400]

    result_df_training, model = make_classifier(result_df_training, top_n, benjamini_hochberg, filename=filename_model, hla_column=hla_column)

    # Make roc curve
    roc_auc = plot_roc_curve(result_df=result_df_training, top_n=top_n, training=True, benjamini_hochberg=benjamini_hochberg, 
                             threshold=threshold, selected_tcrs_patients=selected_tcrs_patients,
                             semi_supervised=semi_supervised, convergence_method=convergence_method, 
                             folder=folder, sort_column=sort_column, hla_column=hla_column)
    sensitivity, specificity, accuracy = calculate_metrics(result_df_training, hla_column=hla_column)

    metrics_training_data = {
            'top_n': top_n, 
            'threshold': threshold,
            'selected_tcrs_patients': selected_tcrs_patients,
            'benjamini_hochberg': benjamini_hochberg,
            'sort_column': sort_column,
            'roc_auc': roc_auc, 
            'sensitivity': sensitivity, 
            'specificity': specificity, 
            'accuracy': accuracy
        }
    
    return model, metrics_training_data
    

def validate_validation_dataset(result_df_validation, top_n=10, number_patients=666, benjamini_hochberg=False, 
                                threshold=None, selected_tcrs_patients=None, model=None, convergence_method=False, 
                                folder='results\\', sort_column='p_value', hla_column='has_HLA_A02_01', semi_supervised=False):
    """
    Validates a dataset using a trained model and computes various evaluation metrics.
    Args:
        result_df_validation (pd.DataFrame): The validation dataset containing features and labels.
        top_n (int, optional): The number of top results to consider for evaluation. Defaults to 10.
        number_patients (int, optional): The number of patients to consider. If 400, the first 400 rows are skipped. Defaults to 666.
        benjamini_hochberg (bool, optional): Whether to apply the Benjamini-Hochberg procedure. Defaults to False.
        threshold (float, optional): The threshold for classification. Defaults to None.
        selected_tcrs_patients (list, optional): A list of selected TCRs for specific patients. Defaults to None.
        model (object, optional): The trained model used for predictions. Must implement `predict_proba` and `predict` methods. Defaults to None.
        convergence_method (bool, optional): Whether to use a convergence method during evaluation. Defaults to False.
        folder (str, optional): The folder path to save results. Defaults to 'results\\'.
        sort_column (str, optional): The column name used for sorting results. Defaults to 'p_value'.
        hla_column (str, optional): The column name representing HLA information. Defaults to 'has_HLA_A02_01'.
        semi_supervised (bool, optional): Whether the evaluation is semi-supervised. Defaults to False.
    Returns:
        dict: A dictionary containing the following validation metrics:
            - 'top_n' (int): The number of top results considered.
            - 'threshold' (float): The classification threshold used.
            - 'selected_tcrs_patients' (list): The selected TCRs for specific patients.
            - 'benjamini_hochberg' (bool): Whether the Benjamini-Hochberg procedure was applied.
            - 'sort_column' (str): The column used for sorting.
            - 'roc_auc' (float): The ROC AUC score.
            - 'sensitivity' (float): The sensitivity of the model.
            - 'specificity' (float): The specificity of the model.
            - 'accuracy' (float): The accuracy of the model.
    """
                                
    if number_patients == 400:
        result_df_validation = result_df_validation.iloc[400:]
    hla_label = hla_column +'_label'
    X_val = result_df_validation[['total_tcrs', 'related_tcrs']]
    y_val = result_df_validation[hla_label]
    result_df_validation['probability'] = model.predict_proba(X_val)[:, 1]
    result_df_validation['prediction'] = model.predict(X_val)
    roc_auc = plot_roc_curve(result_df=result_df_validation, top_n=top_n, training=False, benjamini_hochberg=benjamini_hochberg, 
                             threshold=threshold, selected_tcrs_patients=selected_tcrs_patients, semi_supervised=semi_supervised,
                             convergence_method=convergence_method, folder=folder, sort_column=sort_column, hla_column=hla_column)
    sensitivity, specificity, accuracy = calculate_metrics(result_df_validation, hla_column=hla_column)

    metrics_validation_data = {
        'top_n': top_n, 
        'threshold': threshold,
        'selected_tcrs_patients': selected_tcrs_patients,
        'benjamini_hochberg': benjamini_hochberg,
         'sort_column': sort_column,
        'roc_auc': roc_auc, 
        'sensitivity': sensitivity, 
        'specificity': specificity, 
        'accuracy': accuracy
    }

    return metrics_validation_data




"""
Function to read the HLA labelling of the patients from the Mitchell dataset
"""

def read_patients_hla_labelling_mitchel(hla_path):
    """
    Reads and processes a TSV file containing patient HLA data, filters the data, and identifies patients 
    with the HLA-A*02:01 allele.
    Args:
        hla_path (str): The relative or absolute path to the HLA data file (TSV format).
    Returns:
        pandas.DataFrame: A DataFrame with the following columns:
            - 'repertoire_id': The patient ID.
            - 'has_HLA_A02_01': A boolean indicating whether the patient has the HLA-A*02:01 allele.
    """
    # Read the patients TCR files and make table with related TCRs
    patient_allel_path = path +  hla_path #"\\data\\HLA_Mitchell_Michels_2022.tsv"

    # Lees het TSV-bestand in als DataFrame
    patient_df_mitchell = pd.read_csv(patient_allel_path, sep='\t')


    # Definieer de kolommen die gecontroleerd moeten worden op lege lijsten
    hla_columns = ['HLA-A']#, 'HLA-B', 'HLA-C', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1', 'HLA-DPB1']

    # Filter de DataFrame om alleen rijen te behouden waar niet alle HLA-kolommen '[]' bevatten
    patient_df_filtered = patient_df_mitchell[~patient_df_mitchell[hla_columns].apply(lambda row: all(val == '[]' for val in row), axis=1)]

    #Remove all rows where column 'HLA-A' is NaN
    patient_df_filtered = patient_df_filtered[patient_df_filtered['HLA-A'].notna()]

    # Reset de indexen
    patient_df_filtered.reset_index(drop=True, inplace=True)

    patient_df_filtered = patient_df_filtered[['patient_id', 'HLA-A']]
    patient_df_filtered.rename(columns={'patient_id': 'repertoire_id'}, inplace=True)

    patient_df_filtered['has_HLA_A02_01'] = patient_df_filtered['HLA-A'].apply(lambda x: 'A*02:01' in x if isinstance(x, str) else False)
    patient_df_filtered = patient_df_filtered[['repertoire_id', 'has_HLA_A02_01']] 
    return patient_df_filtered




"""
7. Make the training data imbalanced
# Lower the positive labels for HLA-A 
# (from training data) to find a minimum threshold until where our trained model stays reliable
"""

def select_positive_patients(patient_df, threshold='all'):
    """
    Selects a subset of positive patients based on the specified threshold and combines them 
    with all negative patients.
    Parameters:
    -----------
    patient_df : pandas.DataFrame
        A DataFrame containing patient data. It must include a column named 'has_HLA_A02_01' 
        with boolean values indicating whether a patient is positive (True) or negative (False).
    threshold : int or str, optional, default='all'
        The number of positive patients to retain. If set to 'all', the function returns the 
        original DataFrame without any filtering. If an integer is provided, it specifies the 
        maximum number of positive patients to randomly select.
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing all negative patients and a subset of positive patients based 
        on the specified threshold.
    Notes:
    ------
    - If the threshold is greater than the total number of positive patients, all positive 
      patients will be included.
    - The random selection of positive patients is reproducible due to the fixed random state.
    """

    if threshold == 'all': # Als er geen threshold is, geef de originele DataFrame terug
        return patient_df
    
    
    # Splits positieve en negatieve patiënten
    patient_df_positive = patient_df[patient_df['has_HLA_A02_01'] == True]
    patient_df_negative = patient_df[patient_df['has_HLA_A02_01'] == False]
    
    
    # Willekeurig een subset van de positieve patiënten selecteren op basis van de threshold, 
    # waarbij threshold het aantal patiënten is dat behouden moet worden (tussen 0 en totaal aantal positieve patiënten)
    reduced_positive_df = patient_df_positive.sample(n=min(threshold, len(patient_df_positive)), random_state=42)

    #reduced_positive_df = patient_df_positive.sample(frac=threshold, random_state=42)
    
    # Combineer de negatieve patiënten met de gereduceerde positieve patiënten
    result_df = pd.concat([reduced_positive_df, patient_df_negative], ignore_index=True)
    
    return result_df




def plot_box_plot(result_df, top_n, training=True):
    """
    Plots a boxplot of predicted probabilities grouped by the presence of HLA-A*02:01.
    Args:
        result_df (pd.DataFrame): A DataFrame containing the data to plot. It should have the columns:
            - 'has_HLA_A02_01': Binary column indicating the presence of HLA-A*02:01.
            - 'probability': Column with predicted probabilities.
        top_n (int): The number of top TCRs (T-cell receptors) used in the analysis, included in the plot title.
        training (bool, optional): Indicates whether the data is from the training set or validation set.
            - True: Data is from the training set.
            - False: Data is from the validation set.
            Default is True.
    Returns:
        None: The function generates and displays a boxplot using matplotlib and seaborn.
    Notes:
        - The function customizes the plot title based on the `training` parameter.
        - The plot uses the 'Set2' color palette for visualization.
    """

    # Maak een nieuwe figuur
    plt.figure(figsize=(10, 6))
    
    # Maak een boxplot van de voorspelde kansen
    sns.boxplot(
        x='has_HLA_A02_01',  # Gebruik de kolom 'has_HLA_A02_01' voor de x-as
        y='probability',  # Gebruik de kolom 'probability' voor de y-as
        data=result_df,  # Gebruik de DataFrame met de data
        palette='Set2'  # Gebruik een kleurenpalet
    )
    
    # Voeg labels en titel toe
    plt.xlabel('Has HLA-A*02:01')
    plt.ylabel('Predicted Probability')
    
    if training == True:
            plt.title('Predicted Probability per Patient (Training set, Top ' + str(top_n) + ' TCRs, Mitchel data)')
            #plt.savefig('results\\7\\plots\\Predicted Probability per Patient (Training set, Top ' + str(top_n) + ' TCRs)')
    else:
            plt.title('Predicted Probability per Patient (Validation set, Top ' + str(top_n) + ' TCRs, Mitchel data)')
            #plt.savefig('results\\7\\plots\\Predicted Probability per Patient (Validation set, Top ' + str(top_n) + ' TCRs)')


""""
8. Select top x related TCR's using the convergence method

"""

def read_and_concatenate_files(patients_path, number_of_patients=400):
    """
    Reads and concatenates `.tsv.gz` files from a specified directory, filters the data 
    for rows where the "v_call" column starts with 'TRB', and returns the resulting DataFrame.
    Args:
        patients_path (str): The relative path to the directory containing the patient data files.
        number_of_patients (int, optional): The maximum number of files to process. Defaults to 400.
    Returns:
        pd.DataFrame: A concatenated DataFrame containing filtered data from the processed files.
    Notes:
        - The function assumes that the files are compressed in `.tsv.gz` format.
        - The "chain" column is derived from the first three characters of the "v_call" column.
        - Only rows where the "chain" column equals 'TRB' are included in the final DataFrame.
    """
    data_path = path + patients_path #'\\data\\HLA_emerson_2017'
    all_files = os.listdir(data_path)[:number_of_patients]

    tsv_gz_files = [f for f in all_files if f.endswith('.tsv.gz')]
    data = pd.DataFrame()
    for file in tsv_gz_files:
        file_path = os.path.join(data_path, file)
        with gzip.open(file_path, 'rb') as f:
            df = pd.read_csv(f, sep='\t')
            data = pd.concat([data, df])

    data["chain"] = data["v_call"].str[:3]
    data = data.query("chain == 'TRB'").reset_index(drop=True)

    return data
    

def convergence_method(data, patient_df):
    """
    Perform convergence analysis on TCR (T-cell receptor) data using a TCRDistEmbedder 
    and Fisher's exact test as the convergence metric.
    Args:
        data (pd.DataFrame): The input data containing TCR sequences and associated metadata.
        patient_df (pd.DataFrame): A DataFrame specifying the positive groups (e.g., patient data) 
                                   to compare against the background (negative groups).
    Returns:
        pd.DataFrame: The result of the convergence analysis, containing metrics and embeddings 
                      for the input data.
    """
    tcr_embedder = TCRDistEmbedder(full_tcr=False).fit() # full tcr not needed if grouping by v_call
    fisher = Fisher(
        group_column="repertoire_id", # compare values in the repertoire_id column
        positive_groups=patient_df # compare non-background (positive group) to background (negative group)
    )

    # pass the embedder and the fisher method to the ConvergenceAnalysis object:

    cva = ConvergenceAnalysis(
        tcr_embedder=tcr_embedder,
        convergence_metric=fisher,
        verbose=True,
        index_method="auto"
    )

    cva_res = cva.batched_fit_transform(
        data
    )

    return cva_res


def plot_line_plot(fisher_exact, convergence, compare_metric, base_column='top_n',
                   methods=('_fisher_exact', '_convergence'), sort_column='p_value',
                   folder='results\\', training=True):
    """
    Generates and saves a line plot comparing a specified metric across two methods 
    for different values of a base column.
    Args:
        fisher_exact (pd.DataFrame): DataFrame containing data for the first method.
        convergence (pd.DataFrame): DataFrame containing data for the second method.
        compare_metric (str): The metric to compare between the two methods.
        base_column (str, optional): The column to use as the x-axis. Defaults to 'top_n'.
        methods (tuple, optional): Suffixes for the two methods to distinguish columns 
                                    after merging. Defaults to ('_fisher_exact', '_convergence').
        sort_column (str, optional): The column used for sorting (not directly used in the plot). 
                                        Defaults to 'p_value'.
        folder (str, optional): The base folder where the plot will be saved. Defaults to 'results\\'.
        training (bool, optional): Indicates whether the plot is for the training dataset 
                                    (True) or validation dataset (False). Defaults to True.
    Returns:
        None: The function saves the plot as a PNG file and displays it.
    Notes:
        - The function merges the two input DataFrames on the `base_column` and plots 
            the `compare_metric` for each method.
        - The plot is saved in a subfolder named 'plots' within the specified `folder`.
        - The filename of the saved plot includes the `compare_metric`, `base_column`, 
            `methods`, and whether the dataset is for training or validation.
    """
                   
    
    # Merge the two dataframes on the top_n column and keep the compare_metric column
    merged_df = pd.merge(fisher_exact[[base_column, compare_metric]], convergence[[base_column, compare_metric]], 
                         on=base_column, suffixes=methods)
    
    columns = [compare_metric + methods[i] for i in range(len(methods))]

    plt.figure(figsize=(10, 6))

    for column in columns:
        sns.lineplot(data=merged_df, x=base_column, y=column, label=column, marker='o')

    plt.xlabel('Top N Related TCRs')
    plt.ylabel(compare_metric)
    methods_str = ' and '.join([method[1:] for method in methods])
    if training == True:
        plt.title(f'{compare_metric} per Top N Related TCRs for {methods_str} (sorted on {sort_column}, Training Dataset)')
    elif training == False:
        plt.title(f'{compare_metric} per Top N Related TCRs for {methods_str} (sorted on {sort_column}, Validation Dataset)')
    plt.legend()

    if training == True:
        filename = folder + f'\\plots\\{compare_metric}_per_{base_column}_for_methods_{methods}_sorted_{sort_column}_training_dataset.png'
    elif training == False:
        filename = folder + f'\\plots\\{compare_metric}_per_{base_column}_for_methods_{methods}_sorted_{sort_column}_validation_dataset.png'

    plt.savefig(filename)

    plt.show()
























