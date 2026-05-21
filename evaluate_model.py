import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

def plot_roc_curve(y_true, y_pred_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    print("Loading test data and model...")
    X_test = np.load("X_test.npy").astype(np.float32)
    y_test = np.load("y_test.npy").astype(np.int32)
    
    model = tf.keras.models.load_model("predictive_maintenance_model.keras")
    
    print("\nEvaluating model performance on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    
    # Get predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Extract probabilities for the positive class (Fault)
    y_pred_probs_fault = y_pred_probs[:, 1]
    
    # 1. Classification Report (Text Table)
    print("=== Classification Report ===")
    target_names = ['Normal (0)', 'Fault (1)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 2. Confusion Matrix (Graph)
    print("Generating Confusion Matrix...")
    plot_confusion_matrix(y_test, y_pred, target_names)
    
    # 3. ROC Curve (Graph)
    print("Generating ROC Curve...")
    plot_roc_curve(y_test, y_pred_probs_fault)