# metrics_calculator.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
import os
from sklearn.model_selection import train_test_split
import time
import dcae_sr_eeg_motor_imagery_subpixel as util_pixel
import dcae_sr_eeg_motor_imagery as util_sd
from pandas import DataFrame
from tabulate import tabulate
from tqdm import tqdm

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

# EEG Dataset with synthetic data
class EEGDataset(Dataset):
    def __init__(self, subject_ids, runs, project_path, add_noise=True):
        self.data_lr = []
        self.data_hr = []
        self.add_noise = add_noise
        self.project_path = project_path

        for subject in subject_ids:
            for run in runs:
                orig_sf = 160.0
                duration = 120
                num_samples_orig = int(duration * orig_sf)
                t = np.arange(num_samples_orig) / orig_sf
                data = np.zeros((num_channels, num_samples_orig))
                for ch in range(num_channels):
                    freqs = np.random.uniform(0.5, 70, 5)
                    amps = np.random.uniform(0.5, 2, 5)
                    for f, a in zip(freqs, amps):
                        data[ch] += a * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2*np.pi))
                    data[ch] += np.random.normal(0, 0.5, num_samples_orig)

                sos = signal.butter(4, [0.5 / (orig_sf / 2), 70 / (orig_sf / 2)], 'bandpass', output='sos')
                data = signal.sosfiltfilt(sos, data, axis=1)

                hr_sf = 500.0
                num_samples_hr = int(num_samples_orig * (hr_sf / orig_sf))
                data_hr = signal.resample(data, num_samples_hr, axis=1)
                if data_hr.shape[1] > hr_window_length:
                    data_hr = data_hr[:, :hr_window_length]
                elif data_hr.shape[1] < hr_window_length:
                    padding = np.zeros((num_channels, hr_window_length - data_hr.shape[1]))
                    data_hr = np.hstack((data_hr, padding))

                lr_sf = 50.0
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

        self.data_lr = np.array(self.data_lr)
        self.data_hr = np.array(self.data_hr)
        print(f"Number of segments created: {len(self.data_lr)}")

    def __len__(self):
        return len(self.data_lr)

    def __getitem__(self, idx):
        return torch.tensor(self.data_lr[idx], dtype=torch.float32), torch.tensor(self.data_hr[idx], dtype=torch.float32)

# PSNR, SSIM, and R^2 for 1D signals
def psnr1d(hr, sr):
    mse = np.mean((hr - sr)**2)
    if mse == 0:
        return float('inf')
    max_val = np.max(hr)
    return 10 * np.log10(max_val**2 / mse)

def ssim1d(hr, sr, win_size=11):
    data_range = np.max(hr) - np.min(hr)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    sigma = 1.5
    gauss = signal.windows.gaussian(win_size, sigma)
    gauss /= gauss.sum()

    pad = (win_size - 1) // 2
    hr_padded = np.pad(hr, pad, mode='reflect')
    sr_padded = np.pad(sr, pad, mode='reflect')

    mu_hr = signal.convolve(hr_padded, gauss, mode='valid')
    mu_sr = signal.convolve(sr_padded, gauss, mode='valid')
    mu_hr_sq = mu_hr ** 2
    mu_sr_sq = mu_sr ** 2
    mu_hr_sr = mu_hr * mu_sr

    sigma_hr_sq = signal.convolve(hr_padded ** 2, gauss, mode='valid') - mu_hr_sq
    sigma_sr_sq = signal.convolve(sr_padded ** 2, gauss, mode='valid') - mu_sr_sq
    sigma_hr_sr = signal.convolve(hr_padded * sr_padded, gauss, mode='valid') - mu_hr_sr

    ssim_map = ((2 * mu_hr_sr + c1) * (2 * sigma_hr_sr + c2)) / ((mu_hr_sq + mu_sr_sq + c1) * (sigma_hr_sq + sigma_sr_sq + c2))
    return ssim_map.mean()

def r2_score_1d(hr, sr):
    ss_res = np.sum((hr - sr) ** 2)  # Sum of squared residuals
    mean_hr = np.mean(hr)
    ss_tot = np.sum((hr - mean_hr) ** 2)  # Total sum of squares
    if ss_tot == 0:
        return 0.0  # Avoid division by zero
    return 1 - (ss_res / ss_tot)

# Training function with progress bar
def train_model(model, dataloader, epochs=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for lr_input, hr_target in progress_bar:
            lr_input, hr_target = lr_input.to(device), hr_target.to(device)
            optimizer.zero_grad()
            lr_recon, sr_recon = model(lr_input)
            loss_lr = criterion(lr_recon, lr_input)
            loss_sr = criterion(sr_recon, hr_target)
            loss = loss_lr + loss_sr
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Batch Loss': f'{loss.item():.6f}'})
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.6f}')

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    mse_crit = nn.MSELoss()
    mses = []
    rmses = []
    psnrs = []
    ssims = []
    r2s = []
    inf_times = []

    with torch.no_grad():
        for lr_input, hr_target in dataloader:
            lr_input, hr_target = lr_input.to(device), hr_target.to(device)
            start_time = time.time()
            _, sr_recon = model(lr_input)
            inf_time = time.time() - start_time
            inf_times.append(inf_time)

            mse = mse_crit(sr_recon, hr_target).item()
            rmse = np.sqrt(mse)
            mses.append(mse)
            rmses.append(rmse)

            sr = sr_recon.cpu().numpy()[0]
            hr = hr_target.cpu().numpy()[0]

            psnr_ch = []
            ssim_ch = []
            r2_ch = []
            for ch in range(num_channels):
                hr_ch = hr[ch]
                sr_ch = sr[ch]
                psnr_ch.append(psnr1d(hr_ch, sr_ch))
                ssim_ch.append(ssim1d(hr_ch, sr_ch))
                r2_ch.append(r2_score_1d(hr_ch, sr_ch))
            psnrs.append(np.mean(psnr_ch))
            ssims.append(np.mean(ssim_ch))
            r2s.append(np.mean(r2_ch))

    avg_mse = np.mean(mses)
    avg_rmse = np.mean(rmses)
    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    avg_r2 = np.mean(r2s)
    avg_inf_time = np.mean(inf_times)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'MSE': avg_mse,
        'RMSE': avg_rmse,
        'PSNR': avg_psnr,
        'SSIM': avg_ssim,
        'R^2': avg_r2,
        'Num Parameters': num_params,
        'Avg Inference Time': avg_inf_time
    }

# Main
if __name__ == "__main__":
    subject_ids = list(range(1, 110))
    subjects_train, subjects_test = train_test_split(subject_ids, test_size=0.20, random_state=42)

    train_dataset = EEGDataset(subject_ids=subjects_train, runs=[1, 2], project_path=project_path)
    test_dataset = EEGDataset(subject_ids=subjects_test, runs=[1, 2], project_path=project_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    configs = [
        ("DCAE-SR base", util_sd.DCAE_SR_Base()),
        ("DCAE-SR con PixelShuffle (senza residuale)", util_pixel.DCAE_SR_SubPixel_no_res()),
        ("DCAE-SR con PixelShuffle e residuale", util_pixel.DCAE_SR_SubPixel())
    ]

    results = {}
    print("\n=== Training and Evaluating Models ===")
    for name, model in configs:
        print(f"\nTraining {name}...")
        model = model.to(device)
        train_model(model, train_loader, epochs)
        print(f"Evaluating {name}...")
        results[name] = evaluate_model(model, test_loader)

    # Create DataFrame
    df = DataFrame(results).T  # Transpose to have models as rows

    # Round numerical values for better readability
    df[['MSE', 'RMSE', 'PSNR', 'SSIM', 'R^2']] = df[['MSE', 'RMSE', 'PSNR', 'SSIM', 'R^2']].round(4)
    df['Avg Inference Time'] = df['Avg Inference Time'].round(6)

    # Styling the DataFrame
    def highlight_best(s):
        is_min = s == s.min()
        is_max = s == s.max()
        styles = []
        for min_val, max_val in zip(is_min, is_max):
            if min_val and s.name in ['MSE', 'RMSE', 'Avg Inference Time']:
                styles.append('background-color: lightgreen; font-weight: bold')
            elif max_val and s.name in ['PSNR', 'SSIM', 'R^2']:
                styles.append('background-color: lightgreen; font-weight: bold')
            else:
                styles.append('')
        return styles

    styled_df = df.style.apply(highlight_best, subset=['MSE', 'RMSE', 'PSNR', 'SSIM', 'R^2', 'Avg Inference Time'])\
                        .set_caption("Ablation Study Results")\
                        .set_table_styles([
                            {'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold'), ('text-align', 'center')]},
                            {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold'), ('text-align', 'center')]},
                            {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '8px')]}
                        ])

    # Print formatted table using tabulate
    print("\n=== Ablation Study Results ===")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=True, floatfmt=".4f"))

    # Save styled DataFrame to HTML
    html_path = os.path.join(project_path, "ablation_study_results.html")
    styled_df.to_html(html_path)
    print(f"\nResults saved to {html_path}")

    # Display styled DataFrame in console (if in a Jupyter-like environment)
    try:
        from IPython.display import display
        display(styled_df)
    except ImportError:
        print("\nIPython not available, skipping styled DataFrame display.")
