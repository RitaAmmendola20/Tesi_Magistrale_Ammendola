'''
Script che genera un modello addestrato su eegbci preprocessato, e lo salva in formato .pth
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
import mne
import mne.filter
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
#project_path = os.path.dirname(os.path.abspath(__file__))
project_path=r"C:\Users\RITA\OneDrive\Desktop\Tesi_Magistrale_Ammendola"
#data_path = os.path.join(project_path, 'data')
os.makedirs(project_path, exist_ok=True)

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernels, strides, drops):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_ch[0], out_ch[0], kernels[0], strides[0], padding=(kernels[0] // 2))
        self.conv2 = nn.Conv1d(in_ch[1], out_ch[1], kernels[1], strides[1], padding=(kernels[1] // 2))
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(drops[0])
        self.dropout2 = nn.Dropout(drops[1])

    def forward(self, x):
        x = self.dropout1(self.tanh(self.conv1(x)))
        x = self.dropout2(self.tanh(self.conv2(x)))
        return x

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernels, strides, drops, output_paddings=[0, 0]):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.ConvTranspose1d(
            in_ch[0], out_ch[0], kernels[0], strides[0],
            padding=(kernels[0] // 2), output_padding=output_paddings[0]
        )
        self.conv2 = nn.ConvTranspose1d(
            in_ch[1], out_ch[1], kernels[1], strides[1],
            padding=(kernels[1] // 2), output_padding=output_paddings[1]
        )
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(drops[0])
        self.dropout2 = nn.Dropout(drops[1] if drops[1] is not None else 0)

    def forward(self, x):
        x = self.dropout1(self.tanh(self.conv1(x)))
        x = self.dropout2(self.tanh(self.conv2(x))) if self.dropout2.p > 0 else self.tanh(self.conv2(x))
        return x


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
'''
class DCAE_SR_Base(nn.Module):
    def __init__(self, num_channels=64, lr_len=250, hr_len=2500):
        super(DCAE_SR_Base, self).__init__()
        self.num_channels = num_channels
        self.lr_len = lr_len
        self.hr_len = hr_len

        self.encoder1 = EncoderBlock([num_channels, num_channels], [num_channels, 192], [3,3], [1,1], [0.1,0.1])
        self.encoder2 = EncoderBlock([192, 384], [384, 768], [3,3], [1,1], [0.1,0.1])

        self.bottleneck_conv = nn.Conv1d(768, 768, kernel_size=3, padding=1)
        self.bottleneck_norm = nn.BatchNorm1d(768)
        self.bottleneck_act = nn.ReLU(inplace=True)
        self.bottleneck_drop = nn.Dropout(0.2)

        self.decoder1 = nn.Conv1d(768, 384, kernel_size=3, padding=1)
        self.final_conv_lr = nn.Conv1d(384, num_channels, kernel_size=3, padding=1)

        self.decoder2 = nn.Conv1d(768, num_channels, kernel_size=3, padding=1)
        self.final_conv_sr = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        latent = self.encoder2(self.encoder1(x))  # [B, 768, 250]

        # --- LR Branch ---
        lr_recon = self.decoder_lr2(self.decoder_lr1(latent))  # [B, 64, 250]

        # --- SR Branch ---
        sr_recon = self.decoder_sr2(self.decoder_sr1(latent))  # [B, 250, 64] o simile
        sr_recon = F.interpolate(sr_recon, size=2500, mode='linear', align_corners=False)  # [B, 250, 2500]

        # FIX: riportiamo a [B, 64, 2500]
        sr_recon = sr_recon.permute(0, 2, 1)

        print("DEBUG SHAPES:", lr_recon.shape, sr_recon.shape)  # puoi lasciarlo per conferma
        return lr_recon, sr_recon
'''
class DCAE_SR_Base(nn.Module):
    def __init__(self, num_channels=64, lr_len=250, hr_len=2500):
        super(DCAE_SR_Base, self).__init__()
        self.num_channels = num_channels
        self.lr_len = lr_len
        self.hr_len = hr_len

        # --- Encoder ---
        self.encoder1 = EncoderBlock([num_channels, num_channels], [num_channels, 192], [3,3], [1,1], [0.1,0.1])
        self.encoder2 = EncoderBlock([192, 384], [384, 768], [3,3], [1,1], [0.1,0.1])

        # --- Bottleneck ---
        self.bottleneck_conv = nn.Conv1d(768, 768, kernel_size=3, padding=1)
        self.bottleneck_norm = nn.BatchNorm1d(768)
        self.bottleneck_act = nn.ReLU(inplace=True)
        self.bottleneck_drop = nn.Dropout(0.2)

        # --- LR Decoder ---
        self.decoder_lr = nn.Sequential(
            nn.Conv1d(768, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, num_channels, kernel_size=3, padding=1)
        )

        # --- SR Decoder ---
        self.decoder_sr = nn.Sequential(
            nn.Conv1d(768, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # --- Encoder ---
        latent = self.encoder2(self.encoder1(x))  # [B, 768, 250]

        # --- Bottleneck ---
        latent = self.bottleneck_drop(self.bottleneck_act(self.bottleneck_norm(self.bottleneck_conv(latent))))

        # --- LR Branch ---
        lr_recon = self.decoder_lr(latent)  # [B, 64, 250]

        # --- SR Branch ---
        sr_recon = self.decoder_sr(latent)  # [B, 64, 250] (o [B, 250, 64])

        # Se invertito (250, 64) → correggi
        if sr_recon.shape[1] != self.num_channels and sr_recon.shape[2] == self.num_channels:
            sr_recon = sr_recon.permute(0, 2, 1)

        # Interpolazione temporale a 2500
        sr_recon = F.interpolate(sr_recon, size=self.hr_len, mode='linear', align_corners=False)

        #print("DEBUG SHAPES:", lr_recon.shape, sr_recon.shape)
        return lr_recon, sr_recon


# EEG Dataset Class
# ===============================
# EEGDatasetPreprocessed (CORRETTO)
# ===============================
class EEGDataset(Dataset):
    def __init__(self, subject_ids, runs, project_path, add_noise=True, sr_factor=10):
        self.data_lr = []
        self.data_hr = []
        self.add_noise = add_noise
        self.project_path = project_path
        self.sr_factor = sr_factor

        for subject in subject_ids:
            for run in runs:
                local_path = os.path.join(project_path, f'S{subject:03d}R{run:02d}.edf')

                # --- Scarica se non esiste ---
                if not os.path.exists(local_path):
                    print(f"Scarico S{subject:03d}R{run:02d}.edf...")
                    try:
                        eegbci.load_data(subject, [run], path=os.path.dirname(project_path), update_path=True)
                        src = os.path.join(
                            os.path.dirname(project_path), 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0',
                            f'S{subject:03d}', f'S{subject:03d}R{run:02d}.edf'
                        )
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        shutil.move(src, local_path)
                        print(f"Salvato in: {local_path}")
                    except Exception as e:
                        print(f"Errore download: {e}")
                        continue

                # --- Carica e preprocessa ---
                try:
                    raw = mne.io.read_raw_edf(local_path, preload=True, verbose=False)
                    raw = self._preprocess_raw(raw)
                    data = raw.get_data()           # (64, N)
                    sfreq = raw.info['sfreq']       # 160 Hz

                    # --- Upsample a 500 Hz (HR) ---
                    data_hr = mne.filter.resample(data, up=500.0 / sfreq, npad='auto')
                    data_hr = data_hr[:, :120 * 500]  # 120 sec

                    # --- Downsample a 50 Hz (LR) ---
                    data_lr = mne.filter.resample(data_hr, down=10, npad='auto')

                    # --- Segmenta in finestre ---
                    n_lr = data_lr.shape[1]
                    for start in range(0, n_lr - 250 + 1, 250):
                        lr_seg = data_lr[:, start:start + 250].copy()
                        hr_seg = data_hr[:, start * 10:start * 10 + 2500].copy()

                        # Z-score per canale
                        lr_seg = self._zscore_per_channel(lr_seg)
                        hr_seg = self._zscore_per_channel(hr_seg)

                        if add_noise:
                            lr_seg += np.random.normal(0, 0.05, lr_seg.shape)

                        self.data_lr.append(lr_seg)
                        self.data_hr.append(hr_seg)

                except Exception as e:
                    print(f"Errore elaborazione {local_path}: {e}")
                    continue

        self.data_lr = np.array(self.data_lr, dtype=np.float32)
        self.data_hr = np.array(self.data_hr, dtype=np.float32)
        print(f"Number of segments created: {len(self.data_lr)}")

    def _preprocess_raw(self, raw):
        raw.pick_types(eeg=True, eog=False, stim=False)
        raw.filter(1.0, 40.0, fir_design='firwin')
        raw.notch_filter(50.0, fir_design='firwin')
        raw.set_eeg_reference('average')
        return raw

    def _zscore_per_channel(self, data):
        for ch in range(data.shape[0]):
            m, s = data[ch].mean(), data[ch].std()
            data[ch] = (data[ch] - m) / (s + 1e-8)
        return data

    def __len__(self):
        return len(self.data_lr)

    def __getitem__(self, idx):
        return torch.tensor(self.data_lr[idx]), torch.tensor(self.data_hr[idx])

# Training function
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
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}')

# Example usage
# addestramento e salvataggio del modello
if __name__ == '__main__':
    dataset = EEGDataset(subject_ids=[1], runs=[1, 2], project_path=project_path)
    if len(dataset) == 0:
        print("No data loaded. Check dataset creation process.")
        exit(1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DCAE_SR_Base(num_channels=num_channels).to(device)
    #DEBUG
    print("DEBUG")
    test_lr = torch.randn(1, 64, 250).to(device)
    lr_recon, sr_recon = model(test_lr)
    print(lr_recon.shape, sr_recon.shape)
    print("FINE DEBUG")
    #FINE DEBUG
    train_model(model, dataloader, epochs)
    # Rest of the code remains the same

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

    model_path = os.path.join(project_path, 'dcae_sr_eeg_motor_imagery_n1_031125.pth')
    torch.save(model.state_dict(), model_path)
