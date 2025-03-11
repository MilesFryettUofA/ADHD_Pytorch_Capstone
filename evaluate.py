import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from train import MultiStreamFusionModel  # Adjust the import based on what you need from train.py


# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiStreamFusionModel(metadata_input_dim=3, sensor_input_dim=24, hidden_size=16, output_size=1).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()  # Set to evaluation mode

# Define dataset class (reusing your dataset loader)
class EyeTrackingDataset(torch.utils.data.Dataset):
    def __init__(self, csv_folder, max_seq_len=25000):
        self.csv_folder = csv_folder
        self.csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
        self.max_seq_len = max_seq_len  # Set a fixed sequence length

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        file_path = os.path.join(self.csv_folder, csv_file)

        # Load CSV
        data = pd.read_csv(file_path, dtype=str).apply(pd.to_numeric, errors='coerce')
        data = data.fillna(0)  # Handle missing values

        # Extract metadata, sensor data, and target
        metadata = torch.tensor(data.iloc[:, :3].values, dtype=torch.float32)
        sensor_data = torch.tensor(data.iloc[:, 3:-1].values, dtype=torch.float32)
        target = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)

        # ✅ Apply padding/truncation for consistency
        def pad_tensor(tensor, max_len):
            pad_size = max_len - tensor.shape[0]
            if pad_size > 0:
                padding = torch.zeros((pad_size, tensor.shape[1]), dtype=torch.float32)
                tensor = torch.cat([tensor, padding], dim=0)
            return tensor[:max_len]  # Truncate if too long

        metadata = pad_tensor(metadata, self.max_seq_len)
        sensor_data = pad_tensor(sensor_data, self.max_seq_len)
        target = target[:self.max_seq_len]  # Truncate if needed

        return metadata, sensor_data, target

    def __len__(self):
        return len(self.csv_files)

# Load the dataset
csv_folder = "./data"
dataset = EyeTrackingDataset(csv_folder)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # No shuffling for evaluation

# Evaluate the model on all data
print("\n🔹 **Evaluating Model on Full Dataset** 🔹\n")
with torch.no_grad():  # Disable gradient calculations
    for idx, (metadata, sensor_data, target) in enumerate(data_loader):
        metadata, sensor_data, target = metadata.to(device), sensor_data.to(device), target.to(device)

        # Make predictions
        output = model(metadata, sensor_data)

        # Convert tensors to readable values
        target_value = target.cpu().numpy().tolist()
        output_value = output.cpu().numpy().tolist()

        # Print input and output
        print(f"\n📂 **File {idx+1}/{len(dataset)}**")
        print(f"📊 **Metadata Input:** {metadata.cpu().numpy().tolist()[:3]} ...")  # Print first few metadata values
        print(f"📡 **Sensor Data (First 3 rows):** {sensor_data.cpu().numpy().tolist()[:3]} ...")
        print(f"🎯 **Expected ADHD Severity Score:** {target_value}")
        print(f"🔮 **Predicted ADHD Severity Score:** {output_value}")
        print(f"📉 **Difference:** {abs(output_value[0][0] - target_value[0])}")

print("\n✅ **Evaluation Complete!** 🚀")