from Functions import utils as u
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
    make_scorer
)



def scale_and_split(data, target_column, test_size=0.2, random_state=1, use_smote = False):
    """
    Splits the dataset into training and validation sets, and scales the features.

    Parameters:
    - data: pd.DataFrame
    - target_column: str, name of the target column
    - test_size: float, proportion of validation set
    - random_state: int, for reproducibility

    Returns:
    - X_train_scaled, X_val_scaled, y_train, y_val, scaler
    """
    # Separate features and target
    X = data.drop(columns=[target_column]).copy()
    y = data[target_column].copy()

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    #Resample with smote if True
    if use_smote:
        smote = SMOTE(
            #sampling_strategy=0.7,
            random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    #Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, y_train, y_val, scaler


data = u.extract_data(UMP_threshold=-3.0)

#features_and_label = ['Teff', 'NUVDered','logg','bp_rp_dered','UMP_flag']
features_and_label = ['Teff','logg', 'bp_rp_dered', 'colorColorY','UMP_flag']
selected_data = data[features_and_label].copy()

print(f"Num of UMPs: {len(selected_data[selected_data['UMP_flag']==1])}, Num of non-UMPs:{len(selected_data[selected_data['UMP_flag']!=1])}")

print(f'EMP fraction = {len(selected_data[selected_data['UMP_flag']==1])/len(selected_data)}')

x_train, x_val, y_train, y_val,scaler= scale_and_split(selected_data, 'UMP_flag', use_smote=False)



def UMP_predictor_gbt_model(X_train, y_train, X_val, y_val):

    # ---- Base Gradient Boosted Tree Model ----
    model = GradientBoostingClassifier(random_state=1)

    # ---- Hyperparameter grid ----
    param_grid = {
        'n_estimators': [25,50,75,100,150,],
        'learning_rate': [0.01,0.03,0.04,0.05],
        'max_depth': [2, 3, 4, 5],
        'subsample': [0.7, 0.8,0.9, 1.0],
        'min_samples_split': [2, 5,10],
    }


    scorer = make_scorer(precision_score, zero_division=0)

    search = HalvingGridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        factor=3,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    weights = np.where(y_train == 1, 1.0, 2.0)
    best_model.fit(X_train, y_train, sample_weight=weights)

    print("\nBest Hyperparameters:", search.best_params_)

    # ---- Predicted probabilities for positive class ----
    y_scores = best_model.predict_proba(X_val)[:, 1]

    # ---- Compute Precision–Recall & Best Threshold ----
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    precisions = precisions[:-1]
    recalls = recalls[:-1]
    f1_scores = f1_scores[:-1]
    mask = recalls >= 0.2
    candidate_precisions = precisions[mask]
    candidate_thresholds = thresholds[mask]

    best_index = np.argmax(candidate_precisions)
    best_threshold = candidate_thresholds[best_index]
    
    print(f"\nBest Threshold (max F1): {best_threshold:.4f}")
    print(f"Precision at best threshold: {candidate_precisions[best_index]:.4f}")
    print(f"Recall at best threshold: {recalls[mask][best_index]:.4f}")
    print(f"F1-score at best threshold: {f1_scores[mask][best_index]:.4f}")

    # ---- Apply threshold ----
    y_pred = (y_scores >= best_threshold).astype(int)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Best Threshold Applied)")
    plt.show()

    # ---- Precision–Recall Curve ----
    plt.figure(figsize=(7,4))
    plt.plot(recalls, precisions, linewidth=2)
    plt.scatter(recalls[best_index], precisions[best_index],
                color='red', s=80,
                label=f"Best Threshold = {best_threshold:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Hyperparameter performance visualization ----
    results = search.cv_results_
    mean_f1 = results['mean_test_score']

    # Create labels based on (n_estimators, learning_rate, max_depth)
    labels = [
        f"{results['param_n_estimators'][i]}, "
        f"{results['param_learning_rate'][i]}, "
        f"{results['param_max_depth'][i]}"
        for i in range(len(mean_f1))
    ]

    plt.figure(figsize=(12,4))
    plt.plot(labels, mean_f1, marker='o')
    plt.xticks(rotation=75)
    plt.xlabel("(n_estimators, learning_rate, max_depth)")
    plt.ylabel("Cross-Validated F1 Score")
    plt.title("Gradient Boosted Tree Hyperparameter Results")
    plt.grid(True)
    plt.show()

    return best_model, best_threshold


best_model,best_threshold = UMP_predictor_gbt_model(x_train, y_train, x_val, y_val)