import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from train import MultiStreamFusionModel  # Adjust the import based on what you need from train.py


# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiStreamFusionModel(metadata_input_dim=3, sensor_input_dim=24, hidden_size=64, output_size=1).to(device)
model.load_state_dict(torch.load("best_model_All_Data.pth", map_location=device))
model.eval()  # Set to evaluation mode
# 1️⃣ Define a custom dataset
class EyeTrackingDataset(Dataset):
    def __init__(self, csv_folder, max_seq_len=2500):
        self.csv_folder = csv_folder
        self.csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
        self.max_seq_len = max_seq_len  # Set a fixed length for padding


    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        file_path = os.path.join(self.csv_folder, csv_file)
        
        # Load the CSV file
        data = pd.read_csv(file_path, dtype=str).apply(pd.to_numeric, errors='coerce')
        data = data.fillna(0)

        
        # Define different layers (example: first few columns are metadata, middle are sensor data, last is labels)
        metadata = torch.tensor(data.iloc[:, :3].values, dtype=torch.float32).to(device)  # First 3 columns as metadata
        sensor_data = torch.tensor(data.iloc[:, 3:-1].values, dtype=torch.float32).to(device) # Middle columns as sensor data
        target = torch.tensor(data.iloc[:, -1].values[0], dtype=torch.float32).to(device)  # Last column as the label
        


        downsample_factor = 10  # Keep every 10th row
        sensor_data = sensor_data[::downsample_factor, :]  # Downsample along the sequence dimension
        metadata = metadata[::downsample_factor, :]  # Downsample along the sequence dimension

        def pad_tensor(tensor, max_len):
            pad_size = max_len - tensor.shape[0]
            if pad_size > 0:
                padding = torch.zeros((pad_size, tensor.shape[1]), dtype=torch.float32).to(device)
                tensor = torch.cat([tensor, padding], dim=0).to(device)
            return tensor[:max_len]  # Truncate if too long

        metadata = pad_tensor(metadata, self.max_seq_len)
        sensor_data = pad_tensor(sensor_data, self.max_seq_len)

        return metadata, sensor_data, target

    def __len__(self):
        return len(self.csv_files)


def get_data_loader(train_size_percent =0.0, val_size_percent =1.0, batch_size = 1, csv_folder = './data'):
    dataset = EyeTrackingDataset(csv_folder)

    # Define split sizes
    train_size = int(train_size_percent * len(dataset))  # 70% for training
    val_size = int(val_size_percent * len(dataset))   # 15% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining samples for testing
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    print(f"Train size: {len(train_loader)}")
    print(f"Validation size: {len(val_loader)}")
    print(f"Test size: {len(test_loader)}")
    return train_loader, val_loader, test_loader, dataset

def reset_model_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()



if __name__ == "__main__":
    

    train_loader, val_loader, test_loader, dataset = get_data_loader()

    # Model Creation
    sample_metadata, sample_sensor, sample_target = dataset[0]
    metadata_features = sample_metadata.shape[1]
    sensor_features = sample_sensor.shape[1]
    model = MultiStreamFusionModel(metadata_features, sensor_features, hidden_size=64, output_size=1).to(device)
    model.load_state_dict(torch.load("best_model_All_Data{best_val_loss}.pth", map_location=device))


    


    