import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm  # ✅ Import tqdm for progress bar
import time



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
        
        # Sensor branch: Time-Series Transformer
        self.sensor_linear = nn.Linear(sensor_input_dim, hidden_size)  # Project sensor data to hidden dim

        # Positional Encoding (Learnable for Time-Series)
        self.positional_encoding = nn.Parameter(torch.randn(1, 25000, hidden_size))  # Max sequence length assumed 25000

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=0.2)
        self.sensor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        
        # Fusion layers: combine outputs from both branches
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Ensure output is between 0 and 1

        )
    
    def forward(self, metadata, sensor_data):
        # metadata: [batch_size, num_rows_metadata, metadata_input_dim]
        # sensor_data: [batch_size, seq_len, sensor_input_dim]
        # For metadata, average over rows to get a fixed-size representation
        metadata_avg = metadata.mean(dim=1)  # [batch_size, metadata_input_dim]
        metadata_features = self.metadata_branch(metadata_avg)  # [batch_size, hidden_size]
        

        # Sensor Transformer Processing
        sensor_proj = self.sensor_linear(sensor_data)  
        sensor_proj = sensor_proj + self.positional_encoding[:, :sensor_proj.shape[1], :]  # Add positional encoding
        sensor_proj = sensor_proj.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        
        seq_len = sensor_proj.shape[0]  # Get current sequence length
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(sensor_proj.device)
        mask = mask.masked_fill(mask == 1, float('-inf'))  # Convert to -inf for Transformer masking

        
        sensor_encoded = self.sensor_transformer(sensor_proj, mask)

        # Aggregate information (mean pooling across sequence)
        sensor_encoded = sensor_encoded.mean(dim=0)
        
        # Fuse features from both branches
        fused = torch.cat([metadata_features, sensor_encoded], dim=1)  # [batch_size, hidden_size * 2]
        out = self.fusion(fused)  # [batch_size, output_size]
        return out

# 1️⃣ Define a custom dataset
class EyeTrackingDataset(Dataset):
    def __init__(self, csv_folder, max_seq_len=25000):
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


def get_data_loader(train_size_percent =0.75, val_size_percent =0.25, batch_size = 1, csv_folder = './data'):
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



def train(model, train_loader, val_loader, num_epochs = 5):
    #reset_model_weights(model)  # Reset weights before training

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()  # Start time tracking

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        model.train()

        for batch_idx, (metadata, sensor_data, target) in enumerate(progress_bar):            # Aggregate target values across rows (e.g., average) to get a single target per CSV sample
            
            optimizer.zero_grad(set_to_none=True)
            if(debug):
                print(f"Metadata shape: {metadata.shape}")
                print(f"Sensor data shape: {sensor_data.shape}")
                print(f"Target shape: {target.shape}")
            downsample_factor = 10  # Keep every 10th row
            # Check target shape before taking mean
            if target.dim() > 1:
                target_avg = target.mean(dim=1, keepdim=True)  # Shape: [batch_size, 1]
            else:
                target_avg = target.view(-1, 1)  # Ensure [batch_size, 1] shape  # Shape: [1, 1]


            metadata = metadata[:, ::downsample_factor, :]
            sensor_data = sensor_data[:, ::downsample_factor, :]
            torch.cuda.empty_cache()  # Free unused memory

            output = model(metadata, sensor_data)
            loss = criterion(output, target_avg) # Was target_avg
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            #  Estimate remaining time
            elapsed_time = time.time() - start_time
            batches_done = batch_idx + 1
            avg_batch_time = elapsed_time / batches_done
            remaining_time = avg_batch_time * (len(train_loader) - batches_done)

            #  Update tqdm progress bar with estimated remaining time
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", eta=f"{remaining_time:.2f}s")
        
        avg_train_loss = epoch_loss / len(train_loader)  # Compute average training loss

        # Validation 
        model.eval()
        total_val_loss = 0
        with torch.no_grad():  # No gradient updates during validation
            for metadata, sensor_data, target in val_loader:
                torch.cuda.empty_cache()  # Free unused memory
                output = model(metadata, sensor_data)
                print(f"ADHD Severity score Expected Vs Output:{target.tolist() } | {output.tolist() }")
                if target.dim() > 1:
                    target_avg = target.mean(dim=1, keepdim=True)  # Shape: [batch_size, 1]
                else:
                    target_avg = target.view(-1, 1)  # Ensure [batch_size, 1] shape  # Shape: [1, 1]
                loss = criterion(output, target_avg)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)  # Compute average validation loss
        #  Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_TS.pth")  # Save model weights
            print(f" Model saved at epoch {epoch+1} with validation loss: {avg_val_loss:.4f}")



        # Print results for this epoch
        end_time = time.time()
        epoch_time = end_time - start_time

        print(f" Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds, Avg Loss: {avg_train_loss:.4f}")

if __name__ == "__main__":

    train_loader, val_loader, test_loader, dataset = get_data_loader()

    # Model Creation
    sample_metadata, sample_sensor, sample_target = dataset[0]
    metadata_features = sample_metadata.shape[1]
    sensor_features = sample_sensor.shape[1]
    model = MultiStreamFusionModel(metadata_features, sensor_features, hidden_size=16, output_size=1).to(device)
    #model.load_state_dict(torch.load("best_model.pth", map_location=device))


    train(model, train_loader, val_loader, num_epochs = 500)


    