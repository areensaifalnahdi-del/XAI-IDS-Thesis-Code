import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, auc, precision_recall_curve, roc_auc_score
)
from sklearn.calibration import calibration_curve
import shap
import warnings
import os
 
warnings.filterwarnings('ignore')
 
# ============================================================================
# CONFIGURATION
# ============================================================================
 
TRAIN_FILE = 'CIC_IDS2017_training-set_3000_shuffled.xlsx'
TEST_FILE = 'CIC_IDS2017_testing-set_3000_shuffled.xlsx'
OUTPUT_DIR = './'
 
RF_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': -1,
    'max_depth': 20
}
 
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
 
def load_data():
    """Load training and testing datasets."""
    print(f"Loading training data from {TRAIN_FILE}...")
    train_df = pd.read_excel(TRAIN_FILE)
    
    print(f"Loading testing data from {TEST_FILE}...")
    test_df = pd.read_excel(TEST_FILE)
    
    print(f"✓ Training set shape: {train_df.shape}")
    print(f"✓ Testing set shape: {test_df.shape}")
    
    return train_df, test_df
 
def preprocess_data(train_df, test_df):
    """Preprocess data for machine learning."""
    print("\nPreprocessing data...")
    
    # Extract features and labels
    X_train = train_df.drop(['label', 'id'], axis=1, errors='ignore')
    y_train = train_df['label']
    
    X_test = test_df.drop(['label', 'id'], axis=1, errors='ignore')
    y_test = test_df['label']
    
    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"Encoding categorical columns: {categorical_cols}")
    
    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        combined_data = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(combined_data)
        
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        
    return X_train, X_test, y_train, y_test
 
# ============================================================================
# EVALUATION & METRICS
# ============================================================================
 
def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all performance metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    balanced_accuracy = (recall + (tn / (tn + fp))) / 2 if (tn + fp) > 0 else recall
    brier_score = np.mean((y_pred_proba - y_true) ** 2)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'FPR': fpr,
        'FNR': fnr,
        'Balanced Accuracy': balanced_accuracy,
        'PR-AUC': pr_auc,
        'ROC-AUC': roc_auc,
        'Brier Score': brier_score,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    }
    
    return metrics
 
def save_metrics_table(metrics, output_dir=OUTPUT_DIR):
    """Save metrics to CSV and print."""
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS - CIC-IDS2017")
    print("=" * 80)
    
    # Print metrics
    for key, value in metrics.items():
        if key not in ['TN', 'FP', 'FN', 'TP']:
            print(f"{key:<20}: {value:.4f}")
            
    print(f"\nConfusion Matrix:")
    print(f"TN: {metrics['TN']:<10} FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']:<10} TP: {metrics['TP']}")
    
    # Save to CSV
    metrics_df = pd.DataFrame([
        {'Metric': k, 'Value': v} for k, v in metrics.items()
    ])
    
    output_file = os.path.join(output_dir, "CIC_IDS2017_Metrics.csv")
    metrics_df.to_csv(output_file, index=False)
    print(f"\n✓ Metrics saved to: {output_file}")
 
# ============================================================================
# VISUALIZATIONS
# ============================================================================
 
def plot_confusion_matrix(y_true, y_pred, output_dir=OUTPUT_DIR):
    """Generate confusion matrix visualization."""
    print("Generating Confusion Matrix...")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - CIC-IDS2017', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Benign', 'Attack'], fontsize=12)
    plt.yticks(tick_marks, ['Benign', 'Attack'], fontsize=12)
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight='bold')
            
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "CIC_IDS2017_Confusion_Matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
 
def plot_pr_curve(y_true, y_pred_proba, pr_auc, output_dir=OUTPUT_DIR):
    """Generate Precision-Recall curve."""
    print("Generating PR-AUC Curve...")
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.fill_between(recall, precision, alpha=0.2, color='darkorange')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curve - CIC-IDS2017', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "CIC_IDS2017_PR_Curve.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
 
def plot_calibration_curve(y_true, y_pred_proba, brier_score, output_dir=OUTPUT_DIR):
    """Generate Calibration curve."""
    print("Generating Calibration Curve...")
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, marker='o', color='blue', lw=2,
             label=f'Random Forest (Brier = {brier_score:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    plt.title('Calibration Curve - CIC-IDS2017', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "CIC_IDS2017_Calibration_Curve.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
 
# ============================================================================
# SHAP EXPLANATIONS
# ============================================================================
 
def generate_shap_explanations(model, X_test, y_test, output_dir=OUTPUT_DIR):
    """Generate comprehensive SHAP explanations."""
    print("\n" + "=" * 80)
    print("GENERATING SHAP EXPLANATIONS")
    print("=" * 80)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_attack = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_attack = shap_values[:, :, 1]
    else:
        shap_attack = shap_values
        
    # Export Feature Importance Ranking Table
    print("Exporting Feature Importance Table...")
    vals = np.abs(shap_attack).mean(axis=0)
    if hasattr(vals, "ndim") and vals.ndim > 1:
        vals = vals.mean(axis=-1)
    vals = np.array(vals).flatten()
    
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_Abs_SHAP': vals
    })
    feature_importance.sort_values(by='Mean_Abs_SHAP', ascending=False, inplace=True)
    feature_importance['Rank'] = range(1, len(feature_importance) + 1)
    
    output_csv = os.path.join(output_dir, "CIC_IDS2017_Feature_Ranking.csv")
    feature_importance.to_csv(output_csv, index=False)
    
    # SHAP Bar Plot
    print("Generating SHAP Bar Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_attack, X_test, plot_type="bar", max_display=15, show=False)
    plt.title('SHAP Feature Importance (Top 15) - CIC-IDS2017', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_bar = os.path.join(output_dir, "CIC_IDS2017_SHAP_Bar.png")
    plt.savefig(output_bar, dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP Beeswarm Plot
    print("Generating SHAP Beeswarm Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_attack, X_test, max_display=15, show=False)
    plt.title('SHAP Beeswarm Plot (Top 15) - CIC-IDS2017', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_bee = os.path.join(output_dir, "CIC_IDS2017_SHAP_Beeswarm.png")
    plt.savefig(output_bee, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ All SHAP explanations generated successfully")
 
# ============================================================================
# MAIN EXECUTION
# ============================================================================
 
def main():
    print("\n" + "=" * 80)
    print("CIC-IDS2017 COMPREHENSIVE ANALYSIS PIPELINE")
    print("=" * 80)
    
    try:
        # 1. Load Data
        train_df, test_df = load_data()
        
        # 2. Preprocess Data
        X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)
        
        # 3. Train Model
        print("\nTraining Random Forest model...")
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)
        
        # 4. Make Predictions
        print("Making predictions on test set...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 5. Calculate Metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        save_metrics_table(metrics)
        
        # 6. Generate Visualizations
        print("\nGenerating Performance Visualizations...")
        plot_confusion_matrix(y_test, y_pred)
        plot_pr_curve(y_test, y_pred_proba, metrics['PR-AUC'])
        plot_calibration_curve(y_test, y_pred_proba, metrics['Brier Score'])
        
        # 7. Generate SHAP Explanations
        generate_shap_explanations(model, X_test, y_test)
        
        print("\n" + "=" * 80)
        print("✓ CIC-IDS2017 ANALYSIS COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
 
if __name__ == "__main__":
    main()
