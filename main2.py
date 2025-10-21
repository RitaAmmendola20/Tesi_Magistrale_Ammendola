import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
import mne
from mne.datasets import eegbci
import matplotlib.pyplot as plt
import os
import shutil

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_channels = 64
lr_window_length = 250
hr_window_length = 2500
batch_size = 1
epochs = 20
learning_rate = 0.0001

# Define local data path
project_path = r"C:\Users\RITA\OneDrive\Desktop\Tesi_Magistrale_Ammendola"
os.makedirs(project_path, exist_ok=True)

# -------------------------------
# Blocco SubPixel 1D
# -------------------------------
class SubPixel1D(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(SubPixel1D, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv1d(in_channels, out_channels * upscale_factor, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, C, T]
        x = self.conv(x)  # [B, out_channels*upscale, T]
        B, C, T = x.size()
        r = self.upscale_factor
        if C % r != 0:
            raise ValueError(f"Channel dimension {C} not divisible by upscale factor {r}")
        x = x.view(B, C // r, r, T)       # [B, out_channels, r, T]
        x = x.permute(0, 1, 3, 2)         # [B, out_channels, T, r]
        x = x.contiguous().view(B, C // r, T * r)  # [B, out_channels, T*r]
        return x

# -------------------------------
# Encoder Block
# -------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernels, strides, drops):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_ch[0], out_ch[0], kernels[0], strides[0], padding=kernels[0]//2)
        self.conv2 = nn.Conv1d(in_ch[1], out_ch[1], kernels[1], strides[1], padding=kernels[1]//2)
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(drops[0])
        self.dropout2 = nn.Dropout(drops[1])

    def forward(self, x):
        x = self.dropout1(self.tanh(self.conv1(x)))
        x = self.dropout2(self.tanh(self.conv2(x)))
        return x

# -------------------------------
# DCAE_SR_SubPixel finale
# -------------------------------
class DCAE_SR_SubPixel(nn.Module):
    def __init__(self, num_channels=64, lr_len=250, hr_len=2500):
        super(DCAE_SR_SubPixel, self).__init__()
        self.num_channels = num_channels
        self.lr_len = lr_len
        self.hr_len = hr_len

        # --- Encoder ---
        self.encoder1 = EncoderBlock([num_channels, num_channels], [num_channels, 192], [3,3], [1,1], [0.1,0.1])
        self.encoder2 = EncoderBlock([192, 384], [384, 768], [3,3], [1,1], [0.1,0.1])

        # --- Bottleneck (residual interno) ---
        self.bottleneck_conv = nn.Conv1d(768, 768, kernel_size=3, padding=1)
        self.bottleneck_norm = nn.BatchNorm1d(768)
        self.bottleneck_act = nn.ReLU(inplace=True)
        self.bottleneck_drop = nn.Dropout(0.2)

        # --- Decoder LR ---
        self.decoder1 = nn.Conv1d(768, 384, kernel_size=3, padding=1)
        self.final_conv_lr = nn.Conv1d(384, num_channels, kernel_size=3, padding=1)

        # --- Decoder SR con SubPixel1D ---
        self.decoder2 = nn.Conv1d(768, 640, kernel_size=3, padding=1)  # 64*10 = 640
        self.subpixel = SubPixel1D(in_channels=640, out_channels=64, upscale_factor=10)
        self.final_conv_sr = nn.Conv1d(64, num_channels, kernel_size=3, padding=1)

    def forward(self, input_lr):
        # -------------------------------
        # 1️⃣ Encoder
        # -------------------------------
        x = self.encoder1(input_lr)
        x = self.encoder2(x)

        # -------------------------------
        # 2️⃣ Bottleneck con residual interno
        # -------------------------------
        residual = x
        x = self.bottleneck_conv(x)
        x = self.bottleneck_norm(x)
        x = self.bottleneck_act(x)
        x = self.bottleneck_drop(x)
        x = x + residual  # residual interno

        # -------------------------------
        # 3️⃣ Decoder LR
        # -------------------------------
        x_lr = self.decoder1(x)
        lr_recon = self.final_conv_lr(x_lr)  # [B, C, 250]

        # -------------------------------
        # 4️⃣ Decoder SR
        # -------------------------------
        x_sr = self.decoder2(x)
        x_sr = self.subpixel(x_sr)           # SubPixel 1D upsampling [B, 64, 2500]
        sr_recon = self.final_conv_sr(x_sr)

        # -------------------------------
        # 5️⃣ Residual globale
        # -------------------------------
        input_upsampled = F.interpolate(input_lr, size=self.hr_len, mode='linear', align_corners=False)
        sr_recon = sr_recon + input_upsampled

        return lr_recon, sr_recon


# -----------------------------
# EEG Dataset (unchanged)
# -----------------------------
class EEGDataset(Dataset):
    def __init__(self, subject_ids, runs, project_path, add_noise=True):
        self.data_lr = []
        self.data_hr = []
        self.add_noise = add_noise
        self.project_path = project_path

        for subject in subject_ids:
            for run in runs:
                # Define paths
                download_path = os.path.join(
                    os.path.dirname(project_path), 'MNE-eegbci-data', 'files', 'eegmmidb',
                    '1.0.0', f'S{subject:03d}', f'S{subject:03d}R{run:02d}.edf'
                )
                local_path = os.path.join(project_path, f'S{subject:03d}R{run:02d}.edf')

                # Check if file exists locally
                if not os.path.exists(local_path):
                    print(f"Local file {local_path} not found. Attempting to download...")
                    try:
                        # Download the data if not present
                        eegbci.load_data(subject, [run], path=os.path.dirname(os.path.dirname(project_path)), verbose=True)
                        # Verify that the file was downloaded
                        if not os.path.exists(download_path):
                            print(f"Error: Failed to download {download_path}. Skipping run {run} for subject {subject}.")
                            continue
                        # Move the file to the local path
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        shutil.move(download_path, local_path)
                        print(f"Successfully moved {download_path} to {local_path}")
                    except Exception as e:
                        print(f"Error downloading or moving file for subject {subject}, run {run}: {e}")
                        continue

                # Read and process the EDF file
                try:
                    raw = mne.io.read_raw_edf(local_path, preload=True)
                    raw.filter(0.5, 70, fir_design='firwin')
                    data = raw.get_data()

                    hr_sf = 500
                    orig_sf = raw.info['sfreq']
                    num_samples_hr = int(data.shape[1] * (hr_sf / orig_sf))
                    data_hr = signal.resample(data, num_samples_hr, axis=1)
                    if data_hr.shape[1] > hr_window_length:
                        data_hr = data_hr[:, :hr_window_length]
                    elif data_hr.shape[1] < hr_window_length:
                        padding = np.zeros((num_channels, hr_window_length - data_hr.shape[1]))
                        data_hr = np.hstack((data_hr, padding))

                    lr_sf = 50
                    num_samples_lr = int(hr_window_length * (lr_sf / hr_sf))
                    data_lr = signal.resample(data_hr, num_samples_lr, axis=1)
                    if data_lr.shape[1] > lr_window_length:
                        data_lr = data_lr[:, :lr_window_length]
                    elif data_lr.shape[1] < lr_window_length:
                        padding = np.zeros((num_channels, lr_window_length - data_lr.shape[1]))
                        data_lr = np.hstack((data_lr, padding))

                    if self.add_noise:
                        for ch in range(data_lr.shape[0]):
                            noise = np.random.normal(0, 0.1, data_lr.shape[1])
                            data_lr[ch] += noise

                    for start in range(0, data_lr.shape[1] - lr_window_length + 1, lr_window_length):
                        end_lr = start + lr_window_length
                        end_hr = start + hr_window_length
                        if end_lr <= data_lr.shape[1] and end_hr <= data_hr.shape[1]:
                            self.data_lr.append(data_lr[:, start:end_lr])
                            self.data_hr.append(data_hr[:, start:end_hr])
                except Exception as e:
                    print(f"Error processing file {local_path}: {e}")
                    continue

        self.data_lr = np.array(self.data_lr)
        self.data_hr = np.array(self.data_hr)
        print(f"Number of segments created: {len(self.data_lr)}")

    def __len__(self):
        return len(self.data_lr)

    def __getitem__(self, idx):
        return torch.tensor(self.data_lr[idx], dtype=torch.float32), torch.tensor(self.data_hr[idx], dtype=torch.float32)

# -----------------------------
# Training function (unchanged)
# -----------------------------
def train_model(model, dataloader, epochs=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for lr_input, hr_target in dataloader:
            lr_input, hr_target = lr_input.to(device), hr_target.to(device)
            optimizer.zero_grad()
            lr_recon, sr_recon = model(lr_input)
            loss_lr = criterion(lr_recon, lr_input)
            loss_sr = criterion(sr_recon, hr_target)
            loss = loss_lr + loss_sr
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.6f}')

# -----------------------------
# Example usage
# -----------------------------
if __name__ == '__main__':
    dataset = EEGDataset(subject_ids=[1], runs=[1, 2], project_path=project_path)
    if len(dataset) == 0:
        print("No data loaded. Check dataset creation process.")
        exit(1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DCAE_SR_SubPixel(num_channels=num_channels).to(device)
    train_model(model, dataloader, epochs)

    model.eval()
    test_lr, test_hr = dataset[0]
    test_lr = test_lr.unsqueeze(0).to(device)
    with torch.no_grad():
        _, sr_output = model(test_lr)
    sr_output = sr_output.cpu().numpy()[0]

    plt.figure(figsize=(10, 5))
    plt.plot(sr_output[0], label='SR Output')
    plt.plot(test_hr[0], label='HR Ground Truth', alpha=0.5)
    plt.legend()
    plt.title('SR Output vs HR Ground Truth (Channel 1)')
    plt.xlabel('Time Samples')
    plt.ylabel('Amplitude')
    plt.show()

    model_path = os.path.join(project_path, 'dcae_sr_eeg_motor_imagery_subpixel.pth')
    torch.save(model.state_dict(), model_path)
