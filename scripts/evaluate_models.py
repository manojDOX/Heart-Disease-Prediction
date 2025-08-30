import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
import joblib

graph_dir = "model_graphs"
os.makedirs(graph_dir, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    path = os.path.join(graph_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    path = os.path.join(graph_dir, f"{model_name}_roc_curve.png")
    plt.savefig(path)
    plt.close()
    return path

def evaluate_models(X_test, y_test, model_path='models'):
    model_files = [f for f in os.listdir(model_path) if f.endswith('_model.pkl')]
    results = {}
    models_dict = {}

    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '').capitalize()
        print(f"\n📦 Loading {model_name}...")
        with open(os.path.join(model_path, model_file), 'rb') as f:
            model = pickle.load(f)
            models_dict[model_name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob)

        print(f"📊 Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
        print("📋 Classification Report:")
        print(classification_report(y_test, y_pred))

        plot_confusion_matrix(y_test, y_pred, model_name)
        plot_roc_curve(y_test, y_prob, model_name)

        results[model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc_score
        }

    print("\n✅ All models evaluated!")

    # 🔍 Select Best Model Based on Accuracy (can change to 'roc_auc' or 'f1_score')
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = models_dict[best_model_name]
    best_model_path = os.path.join(model_path, "best_model.pkl")
    joblib.dump(best_model, best_model_path)

    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"✅ Saved best model to: {best_model_path}")

    return results