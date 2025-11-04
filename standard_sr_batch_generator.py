# Initialize dataset and dataloader
from descarded_code import dcae_sr_eeg_motor_imagery as util_sd
import os
import pandas as pd
import matplotlib.pyplot as plt

# Device configuration
device = util_sd.torch.device('cuda' if util_sd.torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_channels = 64
lr_window_length = 250
hr_window_length = 2500
batch_size = 1
epochs = 20
learning_rate = 0.0001

# Define local data path
project_path = r"C:\Users\RITA\OneDrive\Desktop\Tesi_Magistrale_Ammendola"
util_sd.os.makedirs(project_path, exist_ok=True)

# Define output directory
output_dir = util_sd.os.path.join(project_path, 'outputs')
util_sd.os.makedirs(output_dir, exist_ok=True)

subject_ids = [1]  # Example: use subject 1; modify as needed
runs = [1]         # Example: use run 1; modify as needed
dataset = util_sd.EEGDataset(subject_ids=subject_ids, runs=runs, project_path=project_path, add_noise=True)
dataloader = util_sd.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = util_sd.DCAE_SR(num_channels=num_channels)
model = model.to(device)

# Train the model
util_sd.train_model(model, dataloader, epochs=epochs)

# Evaluate and save outputs
model.eval()
with util_sd.torch.no_grad():
    for idx, (lr_input, hr_target) in enumerate(dataloader):
        lr_input, hr_target = lr_input.to(device), hr_target.to(device)
        lr_recon, sr_recon = model(lr_input)

        # Convert outputs to numpy for saving
        lr_input_np = lr_input.cpu().numpy()  # [batch_size, num_channels, lr_window_length]
        lr_recon_np = lr_recon.cpu().numpy()  # [batch_size, num_channels, lr_window_length]
        sr_recon_np = sr_recon.cpu().numpy()  # [batch_size, num_channels, hr_window_length]
        hr_target_np = hr_target.cpu().numpy()  # [batch_size, num_channels, hr_window_length]

        # Save each sample's output
        for b in range(lr_input_np.shape[0]):
            sample_idx = idx * batch_size + b
            sample_output_dir = os.path.join(output_dir, f'sample_sd_{sample_idx}')
            os.makedirs(sample_output_dir, exist_ok=True)

            # Define output types and their data
            output_types = {
                'lr_input_sd': (lr_input_np[b], lr_window_length),
                'lr_recon_sd': (lr_recon_np[b], lr_window_length),
                'sr_recon_sd': (sr_recon_np[b], hr_window_length),
                'hr_target_sd': (hr_target_np[b], hr_window_length)
            }

            # Process each output type
            for output_type, (data, length) in output_types.items():
                output_type_dir = os.path.join(sample_output_dir, output_type)
                os.makedirs(output_type_dir, exist_ok=True)

                # Process each channel
                for ch in range(num_channels):
                    channel_dir = os.path.join(output_type_dir, f'channel_{ch + 1}')
                    os.makedirs(channel_dir, exist_ok=True)

                    # Save as CSV
                    channel_data = data[ch, :length]
                    df = pd.DataFrame(channel_data, columns=['Signal'])
                    csv_path = os.path.join(channel_dir, f'{output_type}_ch{ch + 1}.csv')
                    df.to_csv(csv_path, index=False)

                    # Save as JPG
                    plt.figure(figsize=(10, 4))
                    plt.plot(channel_data, label=f'Channel {ch + 1} ({output_type})')
                    plt.title(f'{output_type} - Channel {ch + 1}')
                    plt.xlabel('Sample Index')
                    plt.ylabel('Amplitude')
                    plt.legend()
                    jpg_path = os.path.join(channel_dir, f'{output_type}_ch{ch + 1}.jpg')
                    plt.savefig(jpg_path)
                    plt.close()

                # Print channel outputs
                print(f'\nSample {sample_idx} - {output_type}:')
                print(f'  Shape: {data.shape}')
                for ch in range(num_channels):
                    print(f'  Channel {ch + 1} (first 10 samples): {data[ch, :10]}')