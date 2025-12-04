# Predicting Neonatal Seizure Risk in HIE Using EEG Metadata, Clinical Features, and a Sarnat Scoring Tool
SAT5141 Final Project Group 3 - Kameron Chung

# Introduction
Hypoxic-Ischaemic Encephalopathy (HIE) is an acute condition that occurs in 1 to 6 births per 1000 live births and is a significant cause of neonatal mortality and morbidity (Bonifacio & Hutson, 2021). This condition is prevalent in late preterm and term infants and most infants who survive have severe neurological impairment lasting their entire lives (Ristovska, Stomnaroska & Danilovski, 2022). HIE is a brain injury that occurs due to an event that occurs during or immediately following birth. Diagnosis for this complex condition involves  multiple factors such as APGAR scoring, umbilical cord gas levels, brain injury, and organ failure. Early detection and diagnosis is extremely important in treating HIE. Despite advances in perinatal care and neuroprotective strategies such as therapeutic hypothermia, HIE remains a major cause of neonatal mortality and long-term neurodevelopmental impairment, including cerebral palsy, cognitive disability and epilepsy (Allen & Brandon, 2011). 

In clinical practice, the Sarnat scoring system is used for neurological assessment of neonates with suspected HIE (Mrelashvili et. al., 2020). The Sarnat exam classifies HIE into three stages, mild, moderate, and severe. Providers will decide to initiate therapeutic hypothermia based on the Sarnat score and other clinical findings. In parallel with clinical presentation, electroencephalography (EEG) plays a central role in the management of neonates with HIE in the Neonatal Intensive Care Unit (NICU). In HIE, abnormal EEG background patterns are strongly associated with brain injury severity and poor outcome (Cornet et. al., 2025). A major contributing factor to the damage of HIE is subclinical seizures (Jain et. al., 2017). Subclinical seizures are seizures that occur without any physical indications and are often only detectable on EEG. Early prediction of seizures using Sarnat scoring in conjunction with EEG features could allow for quick interventions and improved outcomes.

#Project Overview
This project explores clinical decision support for neonatal HIE by building a logistic regression model to predict the possibility of seizures from EEG metadata in term neonates with HIE, building a secondary model using only clinical symptoms, diagnoses, and demographic information, and implementing an interactive Modified Sarnat score calculator providing EEG monitoring recommendation based on neurological exam.

The primary goal of this project is to predict whether a neonate's EEG will show a seizure using EEG metadata and structured features. The project also compares EEG-based prediction to the clinical presentation-based model, demonstrates how a modified Sarnat score can be used to guide clinical intervention, and emphasizes patient-wise splitting and class imbalance handling in small datasets.

This project is NOT a validated clinical tool and is for educational and research purposes ONLY.

# Data Sources

This project uses two open neonatal EEG datasets (automatically loaded from their public URLs):
    
    Neonatal EEG graded for severity of background abnormalities in HIE
      File: [metadata.csv] (https://zenodo.org/records/7477575/files/metadata.csv?download=1)
      Source: Zenodo - https://zenodo.org/records/7477575

      This dataset contains EEG metadata including grade, sampling frequency, epoch number, reference, EEG quality comments, baby_ID, and presence of seizures. This was used for the primary seizure prediction model developed in this project.

    Dataset of Neonatal EEG Recordings with Seizure Annotations
      File: [clinical_information.csv] (https://zenodo.org/records/2547147/files/clinical_information.csv?download=1)
      Source: Zenodo - https://zenodo.org/records/2547147

      This dataset was used as the basis for the Clinical-Only seizure model and for contextual descriptive statistics. This contains multiple clinical variables for each patient and the presence or absence of seizures. 

All files are read directly from their Zenodo URLs in the code. The user does not need to download them manually.

# Environment

This code is intended to run in Jupyter Notebook or Google Colab

Recommended python version: Python 3.8+

Required packages:

    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn ipywidgets

# Code StructureL

All code is contained in one script with three main components:

    1) EEG Metadata Model
    2) Clinical Presentation Model 
    3) Interactive Modified Sarnat Score Calculator

# 1. EEG Metadata Seizure Prediction Model

Data & Preprocessing

    Loads metadata.csv from Zenodo:

    metadata_url = "https://zenodo.org/records/7477575/files/metadata.csv?download=1"
    meta_df = pd.read_csv(metadata_url)


Creates a binary seizure label:

    meta_df['seizure_label'] = meta_df['seizures_YN'].map({'Y': 1, 'N': 0})


Converts key columns to numeric:

    meta_df['grade'] = pd.to_numeric(meta_df['grade'], errors='coerce')
    meta_df['sampling_freq'] = pd.to_numeric(meta_df['sampling_freq'], errors='coerce')


Adds a simple text-based feature: comment length of the EEG quality note:

    meta_df['comment_length'] = meta_df['EEG_quality_comment'].astype(str).str.len()


Defines features and drops rows with missing values in critical predictors:

    feature_cols_numeric = ['grade', 'sampling_freq', 'epoch_number', 'comment_length']
    feature_cols_categorical = ['reference']

    model_df = meta_df.dropna(subset=feature_cols_numeric + ['seizure_label'])
    X = model_df[feature_cols_numeric + feature_cols_categorical]
    y = model_df['seizure_label']
    

Patient-Wise Train/Test Split

  Because the dataset contains EEG segments from the same infants, I identified infants with any seizures vs none using baby_ID and found that there are only 2 babies with seizures in the entire dataset. To avoid data leakage, I enforced 1 seizure baby in the train set, 1 seizure baby in the test set, 20% of non-seizure babies assigned to test, and the rest to train. This ensures that no infant appears in both train and test.
  

Handling Class Imbalance

Seizure epochs are rare. To address this, I oversampled the minority class (seizure) using RandomOverSampler on the training set only:

    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)


The test set is not oversampled, preserving the real-world imbalance for honest evaluation.


Model Architecture

I used a scikit-learn Pipeline with ColumnTransformer for preprocessing, StandardScaler for numeric features, and OneHotEncoder for the EEG reference montage

LogisticRegression classifier:

    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_cols_numeric),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), feature_cols_categorical)
        ]
    )
    
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight=None,  # already using oversampling
        n_jobs=-1,
        random_state=42
    )
    
    clf = Pipeline(steps=[
        ('preprocess', preprocess),
        ('model', log_reg)
    ])
    
    clf.fit(X_train_res, y_train_res)

Evaluation at Default Threshold (0.5)

Predicted probabilities:

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)


Metrics:

    Classification report (precision, recall, F1)

    Confusion matrix heatmap

    ROC-AUC

These metrics give a baseline of model performance at the standard 0.5 decision threshold.

Threshold Tuning to Reduce False Negatives

In neonatal seizure prediction, false negatives (missed seizures) are clinically critical. To address this, the code computes sensitivity (recall) for the positive class across thresholds from 0.0 to 1.0:

    thresholds = np.linspace(0, 1, 101)
    sensitivities = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        sens = recall_score(y_test, y_pred_t, zero_division=0)
        sensitivities.append(sens)
I also plotted sensitivity vs threshold to visualize the trade-off, selected a high-sensitivity operating point with a desired_recall of 0.90 and ensured at least one predicted seizure at that threshold.

If 90% recall is not achievable, the code chooses the best threshold that predicts ≥1 seizure, and maximizes recall.

The code then evaluates the model again at this clinically motivated threshold, showing a new classification report, new confusion matrix, and the ROC-AUC (unchanged, as AUC is threshold-independent)

This demonstrates how a threshold can be adjusted to prioritize seizure detection, accepting more false positives in exchange for fewer missed seizures.

# 2. Clinical Seizure Model (clinical_information.csv)

This secondary model uses clinical and imaging data only.

Data & Preprocessing

Load clinical data:

    clinical_url = "https://zenodo.org/records/2547147/files/clinical_information.csv?download=1"
    clinical_df = pd.read_csv(clinical_url)


Create a per-patient seizure label:

    clinical_df["seizure_label"] = (
        clinical_df["Number of Reviewers Annotating Seizure"] > 0
    ).astype(int)


Drop ID-like and label columns from features:

    drop_cols = ["ID", "EEG file", "Number of Reviewers Annotating Seizure",
                 "Primary Localisation", "Other"]
    model_df = clinical_df.drop(columns=[c for c in drop_cols if c in clinical_df.columns])
    
    X = model_df.drop(columns=["seizure_label"])
    y = model_df["seizure_label"]


One-hot encode selected categorical columns and fill missing values with the column median:

    categorical_cols = ["Gender", "BW (g)", "GA (weeks)", "EEG to PMA (weeks)",
                        "Diagnosis", "Neuroimaging Findings", "PNA of Imaging (days)"]
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_encoded = X_encoded.fillna(X_encoded.median())

Model & Evaluation

Train/test split with stratification:

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )


Standard scaling and logistic regression with class weights:

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )
    log_reg.fit(X_train_scaled, y_train)


Metrics:

    Classification report
    Confusion matrix
    ROC-AUC

Feature importance analysis extracts coefficients from the logistic regression model and ranks features by absolute coefficient magnitude.

This allows the user to visualize the top 15 clinical features associated with seizure vs no seizure.

In practice, this model showed limited discriminative performance (ROC-AUC near chance), reflecting that clinical variables alone are not sufficient for accurate seizure prediction and that EEG features are essential.

# 3. Modified Sarnat Score Calculator (Interactive UI)

The last section implements a bedside-style Modified Sarnat calculator using ipywidgets.

Functionality

    Six Sarnat domains:
      Level of consciousness
      Spontaneous activity
      Posture
      Muscle tone
      Primitive reflexes (suck/Moro)
      Autonomic reflexes (pupils/breathing)

Each domain is scored from 0 to 3 with dropdown options that include clinical descriptions.

The calculator sums the six scores to give the total Sarnat score (0–18) and assigns an HIE stage:

    0–6 → Stage I (Mild HIE)
    7–12 → Stage II (Moderate HIE)
    13–18 → Stage III (Severe HIE)

and Prints:

Numeric scores per category, total score and stage, interpretation and an educational EEG monitoring recommendation.

Usage

In a colab cell:

    sarnat_score_widget()


A form appears with dropdowns and a “Calculate Sarnat Score” button. The result is printed below the widget.

# Interpretation & Limitations

There is a very small number of seizure infants (only 2 in the metadata set), which makes robust model training and validation difficult and forces a custom patient-wise split with 1 seizure infant in train and 1 in test.

Oversampling + threshold tuning are used to prioritize seizure detection over specificity in order to account for severe class imbalance.

Clinical model limitations include clinical variables alone show limited predictive value for seizure occurrence, which reinforces the need for EEG-based features in neonatal seizure prediction.

The models and Sarnat calculator are intended for didactic demonstration and should not replace clinical judgement, EEG interpretation, or established NICU protocols.

# Recorded Presentation

Please follow this link to watch my recorded presentation. 

https://drive.google.com/file/d/1oHJce-2ybXioOqlwpRtnFgkdQfUm3Re1/view
