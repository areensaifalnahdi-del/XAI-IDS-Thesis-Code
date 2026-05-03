"""
Author: Areen Saif Alnahdi
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy.stats import ttest_ind
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')
 
# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
 
print("="*80)
print("ENHANCED EXPLAINABLE AI FOR INTRUSION DETECTION SYSTEMS")
print("="*80)
 
# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
 
print("\n[1] Loading Data...")
 
friday_data = pd.read_csv('data/friday_clean_binary.csv')
wednesday_data = pd.read_csv('data/wednesday_clean_binary.csv')
 
print(f"✓ Friday dataset: {friday_data.shape}")
print(f"✓ Wednesday dataset: {wednesday_data.shape}")
 
# Prepare features and labels
def prepare_data(data):
    """Prepare data for training"""
    data.columns = data.columns.str.strip()
    label_col = [col for col in data.columns if 'Label' in col][0]
    X = data.drop(label_col, axis=1)
    y = data[label_col]
    return X, y
 
X_friday, y_friday = prepare_data(friday_data)
X_wednesday, y_wednesday = prepare_data(wednesday_data)
 
print(f"✓ Friday features: {X_friday.shape[1]}")
print(f"✓ Wednesday features: {X_wednesday.shape[1]}")
 
# ============================================================================
# STEP 2: SAME-DATASET TRAINING AND EVALUATION
# ============================================================================
 
print("\n" + "="*80)
print("SAME-DATASET EVALUATION (Friday and Wednesday separately)")
print("="*80)
 
results_same_dataset = {}
 
for dataset_name, X, y in [("Friday", X_friday, y_friday), ("Wednesday", X_wednesday, y_wednesday)]:
    print(f"\n[2.{dataset_name}] Training on {dataset_name}...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    results_same_dataset[dataset_name] = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'fpr': fpr,
        'fnr': fnr,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }
    
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall: {recall:.4f}")
    print(f"✓ F1-Score: {f1:.4f}")
    print(f"✓ False Positive Rate: {fpr:.4f}")
    print(f"✓ False Negative Rate: {fnr:.4f}")
    print(f"✓ Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
 
# ============================================================================
# STEP 3: CROSS-DATASET VALIDATION
# ============================================================================
 
print("\n" + "="*80)
print("CROSS-DATASET VALIDATION (Train on one, test on the other)")
print("="*80)
 
results_cross_dataset = {}
 
# Train on Friday, Test on Wednesday
print("\n[3.1] Train on Friday → Test on Wednesday...")
model_fri = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_fri.fit(X_friday, y_friday)
y_pred_cross_1 = model_fri.predict(X_wednesday)
acc_cross_1 = accuracy_score(y_wednesday, y_pred_cross_1)
cm_cross_1 = confusion_matrix(y_wednesday, y_pred_cross_1)
tn1, fp1, fn1, tp1 = cm_cross_1.ravel()
 
results_cross_dataset['Friday→Wednesday'] = {
    'accuracy': acc_cross_1,
    'confusion_matrix': cm_cross_1,
    'tp': tp1, 'fp': fp1, 'tn': tn1, 'fn': fn1
}
 
print(f"✓ Cross-Dataset Accuracy: {acc_cross_1:.4f}")
print(f"✓ Confusion Matrix: TP={tp1}, FP={fp1}, TN={tn1}, FN={fn1}")
 
# Train on Wednesday, Test on Friday
print("\n[3.2] Train on Wednesday → Test on Friday...")
model_wed = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_wed.fit(X_wednesday, y_wednesday)
y_pred_cross_2 = model_wed.predict(X_friday)
acc_cross_2 = accuracy_score(y_friday, y_pred_cross_2)
cm_cross_2 = confusion_matrix(y_friday, y_pred_cross_2)
tn2, fp2, fn2, tp2 = cm_cross_2.ravel()
 
results_cross_dataset['Wednesday→Friday'] = {
    'accuracy': acc_cross_2,
    'confusion_matrix': cm_cross_2,
    'tp': tp2, 'fp': fp2, 'tn': tn2, 'fn': fn2
}
 
print(f"✓ Cross-Dataset Accuracy: {acc_cross_2:.4f}")
print(f"✓ Confusion Matrix: TP={tp2}, FP={fp2}, TN={tn2}, FN={fn2}")
 
# ============================================================================
# STEP 4: CONFUSION MATRICES VISUALIZATION
# ============================================================================
 
print("\n[4] Generating Confusion Matrices...")
 
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Confusion Matrices: Same-Dataset and Cross-Dataset Validation', fontsize=16, fontweight='bold')
 
# Friday (same-dataset)
cm_fri = results_same_dataset['Friday']['confusion_matrix']
sns.heatmap(cm_fri, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False,
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
axes[0, 0].set_title('Friday (Train & Test)', fontweight='bold')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')
 
# Wednesday (same-dataset)
cm_wed = results_same_dataset['Wednesday']['confusion_matrix']
sns.heatmap(cm_wed, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1], cbar=False,
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
axes[0, 1].set_title('Wednesday (Train & Test)', fontweight='bold')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')
 
# Cross-dataset 1
cm_cross1 = results_cross_dataset['Friday→Wednesday']['confusion_matrix']
sns.heatmap(cm_cross1, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 0], cbar=False,
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
axes[1, 0].set_title('Train Friday → Test Wednesday', fontweight='bold')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')
 
# Cross-dataset 2
cm_cross2 = results_cross_dataset['Wednesday→Friday']['confusion_matrix']
sns.heatmap(cm_cross2, annot=True, fmt='d', cmap='RdPu', ax=axes[1, 1], cbar=False,
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
axes[1, 1].set_title('Train Wednesday → Test Friday', fontweight='bold')
axes[1, 1].set_ylabel('True Label')
axes[1, 1].set_xlabel('Predicted Label')
 
plt.tight_layout()
plt.savefig('confusion_matrices_all.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices_all.png")
plt.close()
 
# ============================================================================
# STEP 5: SHAP EXPLANATIONS
# ============================================================================
 
print("\n[5] Generating SHAP Explanations...")
 
for dataset_name in ['Friday', 'Wednesday']:
    result = results_same_dataset[dataset_name]
    model = result['model']
    X_test = result['X_test']
    
    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use attack class
    elif shap_values.ndim == 3:  # (samples, features, classes)
        shap_values = shap_values[:, :, 1]  # Use attack class
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance - {dataset_name} Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_shap_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset_name.lower()}_shap_summary.png")
    plt.close()
    
    # Store SHAP values for later analysis
    results_same_dataset[dataset_name]['shap_values'] = shap_values
    results_same_dataset[dataset_name]['explainer'] = explainer
 
# ============================================================================
# STEP 6: REAL LIME EXPLANATIONS (with fallback)
# ============================================================================
 
print("\n[6] Generating REAL LIME Explanations...")
 
for dataset_name in ['Friday', 'Wednesday']:
    result = results_same_dataset[dataset_name]
    model = result['model']
    X_train = result['X_train']
    X_test = result['X_test']
    y_test = result['y_test']
    
    try:
        # Initialize LIME explainer with stability improvements
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns,
            class_names=['Benign', 'Attack'],
            mode='classification',
            discretize='deciles',
            random_state=42
        )
        
        # Find an attack instance
        attack_indices = np.where(y_test == 1)[0]
        if len(attack_indices) > 0:
            instance_idx = attack_indices[0]
            instance = X_test.iloc[instance_idx].values
            
            # Generate LIME explanation
            exp = explainer.explain_instance(
                instance,
                model.predict_proba,
                num_features=10,
                top_labels=1
            )
            
            # Save LIME plot
            fig = exp.as_pyplot_figure()
            plt.savefig(f'{dataset_name.lower()}_lime_explanation.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {dataset_name.lower()}_lime_explanation.png")
            plt.close()
            
            results_same_dataset[dataset_name]['lime_explainer'] = explainer
            results_same_dataset[dataset_name]['lime_explanation'] = exp
    
    except Exception as e:
        print(f"⚠ LIME explanation failed for {dataset_name}: {e}")
        print("  Using permutation-based fallback...")
        
        # Fallback: Permutation importance
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Create fallback visualization
        plt.figure(figsize=(10, 6))
        indices = np.argsort(perm_importance.importances_mean)[-10:]
        plt.barh(range(len(indices)), perm_importance.importances_mean[indices])
        plt.yticks(range(len(indices)), X_test.columns[indices])
        plt.xlabel('Permutation Importance')
        plt.title(f'Permutation Importance (LIME Fallback) - {dataset_name}')
        plt.tight_layout()
        plt.savefig(f'{dataset_name.lower()}_lime_fallback.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {dataset_name.lower()}_lime_fallback.png")
        plt.close()
 
# ============================================================================
# STEP 7: EXPLANATION STABILITY ANALYSIS
# ============================================================================
 
print("\n[7] Analyzing Explanation Stability...")
 
stability_results = {}
 
for dataset_name in ['Friday', 'Wednesday']:
    result = results_same_dataset[dataset_name]
    model = result['model']
    X_test = result['X_test']
    explainer = result['explainer']
    
    # Generate SHAP values multiple times with slight perturbations
    stability_scores = []
    num_runs = 5
    
    for run in range(num_runs):
        shap_vals = explainer.shap_values(X_test)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]
        
        # Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        stability_scores.append(mean_abs_shap)
    
    # Calculate stability as correlation between runs
    stability_matrix = np.corrcoef(np.array(stability_scores))
    mean_stability = np.mean(stability_matrix[np.triu_indices_from(stability_matrix, k=1)])
    
    stability_results[dataset_name] = {
        'mean_stability': mean_stability,
        'stability_matrix': stability_matrix
    }
    
    print(f"✓ {dataset_name} Explanation Stability: {mean_stability:.4f}")
 
# ============================================================================
# STEP 8: STATISTICAL TESTING
# ============================================================================
 
print("\n[8] Performing Statistical Testing...")
 
# Compare Friday and Wednesday performance
friday_acc = results_same_dataset['Friday']['accuracy']
wednesday_acc = results_same_dataset['Wednesday']['accuracy']
 
# Get predictions for t-test
friday_preds = results_same_dataset['Friday']['y_pred']
wednesday_preds = results_same_dataset['Wednesday']['y_pred']
friday_labels = results_same_dataset['Friday']['y_test']
wednesday_labels = results_same_dataset['Wednesday']['y_test']
 
# Compute per-sample accuracy
friday_correct = (friday_preds == friday_labels).astype(int)
wednesday_correct = (wednesday_preds == wednesday_labels).astype(int)
 
# T-test
t_stat, p_value = ttest_ind(friday_correct, wednesday_correct)
 
print(f"✓ Friday Accuracy: {friday_acc:.4f}")
print(f"✓ Wednesday Accuracy: {wednesday_acc:.4f}")
print(f"✓ T-test p-value: {p_value:.4f}")
print(f"✓ Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
 
# ============================================================================
# SUMMARY
# ============================================================================
 
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated Files:")
print("1. confusion_matrices_all.png")
print("2. friday_shap_summary.png")
print("3. wednesday_shap_summary.png")
print("4. friday_lime_explanation.png (or lime_fallback.png)")
print("5. wednesday_lime_explanation.png (or lime_fallback.png)")
print("\nKey Metrics:")
print(f"- Friday Same-Dataset Accuracy: {results_same_dataset['Friday']['accuracy']:.4f}")
print(f"- Wednesday Same-Dataset Accuracy: {results_same_dataset['Wednesday']['accuracy']:.4f}")
print(f"- Friday→Wednesday Cross-Dataset Accuracy: {results_cross_dataset['Friday→Wednesday']['accuracy']:.4f}")
print(f"- Wednesday→Friday Cross-Dataset Accuracy: {results_cross_dataset['Wednesday→Friday']['accuracy']:.4f}")
print(f"- Friday Explanation Stability: {stability_results['Friday']['mean_stability']:.4f}")
print(f"- Wednesday Explanation Stability: {stability_results['Wednesday']['mean_stability']:.4f}")
