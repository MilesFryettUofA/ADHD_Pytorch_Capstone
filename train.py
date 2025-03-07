import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.amp as amp

from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import gc

gc.collect()  # Force garbage collection
debug = False
torch.cuda.empty_cache()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model again (from our previous discussion)
class MultiStreamFusionModel(nn.Module):
    def __init__(self, metadata_input_dim, sensor_input_dim, hidden_size, output_size, nhead=4, num_transformer_layers=2):
        super(MultiStreamFusionModel, self).__init__()
        
        # Metadata branch: simple MLP
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Sensor branch: Transformer-based branch
        # First, project sensor features to hidden_size
        self.sensor_linear = nn.Linear(sensor_input_dim, hidden_size)
        # Transformer encoder expects input shape [seq_len, batch_size, d_model]
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.sensor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Fusion layers: combine outputs from both branches
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
                nn.Sigmoid()  # Ensure output is between 0 and 1

        )
    
    def forward(self, metadata, sensor_data):
        # metadata: [batch_size, num_rows_metadata, metadata_input_dim]
        # sensor_data: [batch_size, seq_len, sensor_input_dim]
        # For metadata, average over rows to get a fixed-size representation
        metadata_avg = metadata.mean(dim=1)  # [batch_size, metadata_input_dim]
        metadata_features = self.metadata_branch(metadata_avg)  # [batch_size, hidden_size]
        
        # Process sensor data with the transformer
        # Project sensor data first
        sensor_proj = self.sensor_linear(sensor_data)  # [batch_size, seq_len, hidden_size]
        # Transformer expects [seq_len, batch_size, hidden_size]
        sensor_proj = sensor_proj.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        sensor_encoded = self.sensor_transformer(sensor_proj)  # [seq_len, batch_size, hidden_size]
        # Transpose back
        sensor_encoded = sensor_encoded.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        # Aggregate the sequence dimension (e.g., using mean pooling)
        sensor_features = sensor_encoded.mean(dim=1)  # [batch_size, hidden_size]
        
        # Fuse features from both branches
        fused = torch.cat([metadata_features, sensor_features], dim=1)  # [batch_size, hidden_size * 2]
        out = self.fusion(fused)  # [batch_size, output_size]
        return out

# 1️⃣ Define a custom dataset
class EyeTrackingDataset(Dataset):
    def __init__(self, csv_folder):
        self.csv_folder = csv_folder
        self.csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        file_path = os.path.join(self.csv_folder, csv_file)
        
        # Load the CSV file
        data = pd.read_csv(file_path, dtype=str).apply(pd.to_numeric, errors='coerce')
        data = data.fillna(0)

        
        # Define different layers (example: first few columns are metadata, middle are sensor data, last is labels)
        metadata = torch.tensor(data.iloc[:, :3].values, dtype=torch.float32).to(device)  # First 3 columns as metadata
        sensor_data = torch.tensor(data.iloc[:, 3:-1].values, dtype=torch.float32).to(device) # Middle columns as sensor data
        target = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).to(device)  # Last column as the label
        
        return metadata, sensor_data, target

    def __len__(self):
        return len(self.csv_files)


def get_data_loader(train_sizeP =0.75, val_sizeP =0.20, batch_size = 1, csv_folder = './data'):
    dataset = EyeTrackingDataset(csv_folder)

    # Define split sizes
    train_size = int(train_sizeP * len(dataset))  # 70% for training
    val_size = int(val_sizeP * len(dataset))   # 15% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining samples for testing
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    print(f"Train size: {len(train_loader)}")
    print(f"Validation size: {len(val_loader)}")
    print(f"Test size: {len(test_loader)}")
    return train_loader, val_loader, test_loader, dataset



def train(model, train_loader, val_loader, num_epochs = 5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for metadata, sensor_data, target in train_loader:
            # Aggregate target values across rows (e.g., average) to get a single target per CSV sample
            target_avg = target.mean(dim=1, keepdim=True)  # Shape: [batch_size, 1]
            
            optimizer.zero_grad(set_to_none=True)
            if(debug):
                print(f"Metadata shape: {metadata.shape}")
                print(f"Sensor data shape: {sensor_data.shape}")
                print(f"Target shape: {target.shape}")
            downsample_factor = 5  # Keep every 10th row

            metadata = metadata[:, ::downsample_factor, :]
            sensor_data = sensor_data[:, ::downsample_factor, :]
            target = target[:, ::downsample_factor]
            output = model(metadata, sensor_data)
            loss = criterion(output, target[0].long()) # Was target_avg
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)  # Compute average training loss
        ### VALIDATION PHASE ###
        model.eval()
        total_val_loss = 0
        with torch.no_grad():  # No gradient updates during validation
            for metadata, sensor_data, target in val_loader:
                output = model(metadata, sensor_data)
                print(f"ADHD Severity score Expected Vs Output:{target * 100} | {output * 100} || Differnce:{(output * 100) - (target * 100)}")
                if (output)
                loss = criterion(output, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)  # Compute average validation loss
        # ✅ Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")  # Save model weights
            print(f"✅ Model saved at epoch {epoch+1} with validation loss: {avg_val_loss:.4f}")


        # Print results for this epoch
        print(f"Epoch [{epoch+1}/5], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":

    train_loader, val_loader, test_loader, dataset = get_data_loader()

    # Model Creation
    sample_metadata, sample_sensor, sample_target = dataset[0]
    metadata_features = sample_metadata.shape[1]
    sensor_features = sample_sensor.shape[1]
    model = MultiStreamFusionModel(metadata_features, sensor_features, hidden_size=16, output_size=1).to(device)

    train(model, train_loader, num_epochs = 5)


    