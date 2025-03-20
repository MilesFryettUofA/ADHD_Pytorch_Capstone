import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import numpy as np

# Load your dataset class and model
from train import EyeTrackingDataset, MultiStreamFusionModel  # Update with your actual file name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test dataset
def load_test_data(csv_folder, batch_size=1):
    dataset = EyeTrackingDataset(csv_folder)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_loader, dataset

# Function to evaluate the model
def evaluate_model(model, test_loader, threshold=0.5, find_best_threshold=False):
    model.eval()
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for metadata, sensor_data, target in test_loader:
            metadata, sensor_data, target = metadata.to(device), sensor_data.to(device), target.to(device)

            # Get model output
            output = model(metadata, sensor_data).squeeze().cpu().numpy()
            target = target.squeeze().cpu().numpy()

            # Store predictions and true labels
            y_scores.append(output)
            y_true.append(target)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # ✅ Find the best threshold using ROC curve if needed
    if find_best_threshold:
        y_true = (y_true >= 0.5).astype(int)  # Threshold at 0.5
        y_scores = (y_scores >= 0.7).astype(int)  # Threshold at 0.5

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        optimal_idx = (tpr - fpr).argmax()
        threshold = thresholds[optimal_idx]
        print(f"Optimal Threshold Found: {threshold}")

    # Convert scores to binary predictions
    y_pred = (y_scores >= threshold).astype(int)

    # ✅ Compute Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n🔹 Model Evaluation Results 🔹")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # ✅ Save results to CSV
    results_df = pd.DataFrame({"True_Label": y_true, "Predicted_Probability": y_scores, "Binary_Prediction": y_pred})
    results_df.to_csv("model_predictions.csv", index=False)
    print("✅ Predictions saved to 'model_predictions.csv'")

    return accuracy, precision, recall, f1, threshold

# ✅ Load trained model
def load_trained_model(model_path, dataset):
    metadata_features = dataset[0][0].shape[1]
    sensor_features = dataset[0][1].shape[1]
    
    model = MultiStreamFusionModel(metadata_features, sensor_features, hidden_size=64, output_size=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model

# ✅ Run evaluation
if __name__ == "__main__":
    test_loader, dataset = load_test_data(csv_folder="./data", batch_size=1)
    model = load_trained_model(model_path="best_model_All_Data{best_val_loss}.pth", dataset=dataset)
    evaluate_model(model, test_loader, find_best_threshold=True)
