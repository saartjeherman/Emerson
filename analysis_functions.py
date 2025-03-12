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
    files_path = path + patients_path #'\\data\\HLA_emerson_2017'

    all_files = os.listdir(files_path)

    tsv_gz_files = [f for f in all_files if f.endswith('.tsv.gz')]

    if training == True:
        tsv_gz_files = tsv_gz_files[:number_of_patients] #tsv_gz_files[:400]

    elif training == False:
        tsv_gz_files = tsv_gz_files[number_of_patients:]

    patient_names = [f.split('.')[0] for f in tsv_gz_files]

    return tsv_gz_files, patient_names



def matching_patients_tcrs(patients_path, number_of_patients=400, top_n=None):
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
    tcr_df = tcr_df[tcr_df['repertoire_ids'].apply(len) >= 5]
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

def match_hla(patients_path, hla_path):
    tsv_gz_files, all_patient_names = read_patient_files(patients_path)

    record =  {}  # Temporary dictionary to hold one record
    file_path = path + hla_path #'\\data\\A_02_01_features.txt'

    # Read and parse the file
    with open(file_path, "r") as file:
        for line in file:
            line = line.split(" ")
            if line[1] == 'HLA-A*02:01':
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

    patient_df = pd.DataFrame(list(record.items()), columns=['repertoire_id', 'has_HLA_A02_01'])
    patient_df.sort_values(by=['repertoire_id'], inplace=True, ignore_index=True)
    return patient_df


## select number of patients of the hla labelling for training or testing and remove rows with missing values
def select_patients_hla_labelling(patient_df, number_of_patients=666, hla_column='has_HLA_A02_01', training=True):
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

def execute_fisher_exact(tcr_df, patient_df):
    fisher_exact_results = []
    for index, row in tcr_df.iterrows():
        # Haal de eerste rij van tcr_df en zijn combinatie en repertoire_ids
        repertoire_ids = row['repertoire_ids']  # Aangenomen dat repertoire_ids een lijst is
        combination = row['combination']

        # Filter patient_df op basis van de repertoire_ids
        filtered_patients = patient_df[patient_df['repertoire_id'].isin(repertoire_ids)]

        # Tel het aantal patiënten in elke categorie
        have_hla_and_tcr = len(filtered_patients[filtered_patients['has_HLA_A02_01'] == True]) + 0.1  # Aantal met HLA en TCR
        have_no_hla_and_tcr = len(filtered_patients[filtered_patients['has_HLA_A02_01'] == False]) + 0.1  # Aantal zonder HLA maar met TCR


        # Aantal patiënten zonder TCR
        have_hla_no_tcr = len(patient_df[(patient_df['has_HLA_A02_01'] == True) & 
                                                (~patient_df['repertoire_id'].isin(repertoire_ids))]) + 0.1
        have_no_hla_no_tcr = len(patient_df[(patient_df['has_HLA_A02_01'] == False) & 
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
            'HLA': 'HLA-A*02:01',
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
                                  benjamini_hochberg=False, selected_tcrs_patients=None):

    tsv_gz_files, patient_names = read_patient_files(patients_path, number_patients)

    files_path = path + patients_path #'\\data\\HLA_emerson_2017'

    result_data = []
    related_tcrs = fisher_exact_results_df[(fisher_exact_results_df['p_value'] < 0.05) & (fisher_exact_results_df['odds_ratio'] > 1)].reset_index(drop=True)
    print("Related TCRs fisher exact: ",len(related_tcrs)) 
    top_related_tcrs = related_tcrs.sort_values(by=[sort_column]).head(top_n)

    if benjamini_hochberg:
        fisher_exact_results_df = apply_benjamini_hochberg(fisher_exact_results_df)
        related_tcrs = fisher_exact_results_df[(fisher_exact_results_df['corrected_p_value'] < 0.05) & (fisher_exact_results_df['odds_ratio'] > 1)].reset_index(drop=True)
        if sort_column == 'p_value':
            top_related_tcrs = related_tcrs.sort_values(by=['corrected_p_value']).head(top_n)
        else:
            top_related_tcrs = related_tcrs.sort_values(by=[sort_column]).head(top_n)

    print("Selected related TCRs: ",len(top_related_tcrs)) 

    
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
                continue  # Sla over als de vereiste kolommen ontbreken
            
            # Haal totalen op
            total_tcrs = len(df)
            rep_id = df['repertoire_id'].iloc[0]  # Gebruik .iloc om een fout te voorkomen

            # check if value of HLA-A*02:01 is NaN
            has_hla = None
            hla_row = hla_labelling[hla_labelling['repertoire_id'] == rep_id]
            if not hla_row.empty:
                has_hla = hla_row['has_HLA_A02_01'].iloc[0]
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
                'has_HLA_A02_01': has_hla
            })
            

    # Als je de data als een DataFrame wilt hebben
    result_df = pd.DataFrame(result_data)

    if len(top_related_tcrs) < top_n:
        return result_df, len(top_related_tcrs)

    return result_df 

    
def calculate_metrics(result_df):
    # Compute the confusion matrix
    cm = confusion_matrix(result_df['has_HLA_A02_01_label'], result_df['prediction'])
    TN, FP, FN, TP = cm.ravel()

    # Calculate metrics
    sensitivity = TP / (TP + FN)  # True Positive Rate
    specificity = TN / (TN + FP)  # True Negative Rate
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # Overall accuracy

    # Print results
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    print(classification_report(result_df['has_HLA_A02_01_label'], result_df['prediction'], target_names=['Class 0', 'Class 1']))
    return sensitivity, specificity, accuracy


#make a ROC curve for the trained model
def plot_roc_curve(result_df, top_n, training=True, benjamini_hochberg=False, 
                   threshold=None, selected_tcrs_patients=None, semi_supervised=False):
    # Bereken ROC-curve en AUC
    fpr, tpr, thresholds = roc_curve(result_df['has_HLA_A02_01_label'], result_df['probability'])
    roc_auc = auc(fpr, tpr)

    # Plot instellingen
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Bepaal titel en bestandsnaam
    dataset_type = "training" if training else "validation"
    title = f"ROC Logistic Regression top {top_n} related tcrs ({dataset_type.capitalize()} set)"
    filename = f"results\\plots\\ROC_Logistic_Regression_top{top_n}_related_tcrs_{dataset_type}.png"

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

    # Toon en sla de plot op
    print(title)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.5)
    plt.savefig(filename)
    plt.show()

    return roc_auc



def make_classifier(result_df, top_n=10, benjamini_hochberg=False, filename=None):
    # make classifier logistic regression: 
    # - features are total_tcrs and related_tcrs
    # - target is has_HLA_A02_01 
    # - save the model

    model = LogisticRegression()
    X = result_df[['total_tcrs', 'related_tcrs']]
    y = result_df['has_HLA_A02_01_label']
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
                          top_n=10, benjamini_hochberg=False, selected_tcrs_patients=None):
    print("preprocessing dataset")
    result_df = match_and_select_related_tcrs(fisher_exact_results_df=fisher_exact_results_df, hla_labelling=patient_df, 
                                              patients_path=patients_path, top_n=top_n, benjamini_hochberg=benjamini_hochberg, 
                                              selected_tcrs_patients=selected_tcrs_patients)
    print(result_df)
    # Make a label for the target (has_HLA_A02_01), model can't handle boolean values
    # 0 = False, 1 = True
    result_df['has_HLA_A02_01'] = result_df['has_HLA_A02_01'].replace('None', np.nan)
    #drop none values
    result_df = result_df.dropna()
    result_df['has_HLA_A02_01_label'] = result_df['has_HLA_A02_01'].astype('category').cat.codes
    return result_df


def train_training_dataset(result_df_training, top_n=10, benjamini_hochberg=False, 
                           threshold=None, selected_tcrs_patients=None, filename_model=None,
                           semi_supervised=False):
    metrics_training_data = []
    
    #select first 400 patients for training
    result_df_training.sort_values(by=['repertoire_id'], inplace=True, ignore_index=True)
    if semi_supervised == False:    
        result_df_training = result_df_training.iloc[:400]

    result_df_training, model = make_classifier(result_df_training, top_n, benjamini_hochberg, filename=filename_model)

    # Make roc curve
    roc_auc = plot_roc_curve(result_df_training, top_n, benjamini_hochberg, 
                             threshold=threshold, selected_tcrs_patients=selected_tcrs_patients,
                             semi_supervised=semi_supervised)
    sensitivity, specificity, accuracy = calculate_metrics(result_df_training)

    metrics_training_data.append({
            'top_n': top_n, 
            'threshold': threshold,
            'selected_tcrs_patients': selected_tcrs_patients,
            'benjamini_hochberg': benjamini_hochberg,
            'roc_auc': roc_auc, 
            'sensitivity': sensitivity, 
            'specificity': specificity, 
            'accuracy': accuracy
        })
    
    return model, metrics_training_data
    

def validate_validation_dataset(result_df_validation, top_n=10, number_patients=666, benjamini_hochberg=False, 
                                threshold=None, selected_tcrs_patients=None, model=None):
    metrics_validation_data = []
    if number_patients == 400:
        result_df_validation = result_df_validation.iloc[400:]
    X_val = result_df_validation[['total_tcrs', 'related_tcrs']]
    y_val = result_df_validation['has_HLA_A02_01_label']
    result_df_validation['probability'] = model.predict_proba(X_val)[:, 1]
    result_df_validation['prediction'] = model.predict(X_val)
    roc_auc = plot_roc_curve(result_df_validation, top_n, False, benjamini_hochberg, 
                             threshold=threshold, selected_tcrs_patients=selected_tcrs_patients)
    sensitivity, specificity, accuracy = calculate_metrics(result_df_validation)

    metrics_validation_data.append({
        'top_n': top_n, 
        'threshold': threshold,
        'selected_tcrs_patients': selected_tcrs_patients,
        'benjamini_hochberg': benjamini_hochberg,
        'roc_auc': roc_auc, 
        'sensitivity': sensitivity, 
        'specificity': specificity, 
        'accuracy': accuracy
    })

    return metrics_validation_data






"""
Function to read the HLA labelling of the patients from the Mitchell dataset
"""

def read_patients_hla_labelling_mitchel(hla_path):
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
    Reduceert het aantal positieve labels ('has_HLA_A02_01' == True) tot een bepaalde threshold.
    
    Parameters:
        patient_df (pd.DataFrame): DataFrame met patiënten, inclusief de kolom 'has_HLA_A02_01'.
        threshold (float): Proportie van positieve labels om te behouden (tussen 0 en 1).
        
    Returns:
        pd.DataFrame: Gereduceerde DataFrame met de gewenste proportie positieve labels.
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
    Maakt een boxplot van de voorspelde kansen van de patiënten.
    
    Parameters:
        result_df (pd.DataFrame): DataFrame met de patiënten en hun voorspelde kansen.
        top_n (int): Aantal TCRs waarop de modellen getraind zijn.
        training (bool): Of de patiënten uit de training- of validatieset komen.
        threshold (int): De threshold die gebruikt is om de patiënten te selecteren.
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
    tcr_embedder = TCRDistEmbedder(full_tcr=False).fit() # full tcr not needed if grouping by v_call
    fisher = Fisher(
        group_column="repertoire_id", # compare values in the repertoire_id column
        positive_groups=patient_df # compare non-background (positive group) to background (negative group)
    )

    # pass the embedder and the fisher method to the ConvergenceAnalysis object:

    cva = ConvergenceAnalysis(
        tcr_embedder=tcr_embedder,
        convergence_metric=fisher,
        verbose=True
    )

    cva_res = cva.batched_fit_transform(
        data
    )

    return cva_res


























