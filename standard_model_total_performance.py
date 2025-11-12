import time
import standard_model_generator as util_sd
from pandas import DataFrame
from tabulate import tabulate
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy import signal
import os

# ===============================
# Config
# ===============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_path = r"C:\Users\RITA\OneDrive\Desktop\Tesi_Magistrale_Ammendola"
model_path = os.path.join(project_path, "dcae_sr_eeg_motor_imagery_n1_041125.pth")
num_channels = 64
batch_size = 1

# ===============================
# Metriche 1D
# ===============================
def psnr1d(hr, sr):
    mse = np.mean((hr - sr) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(np.abs(hr))
    return 20 * np.log10(max_val / (np.sqrt(mse) + 1e-10))

def ssim1d(hr, sr, win_size=11):
    data_range = hr.max() - hr.min()
    if data_range == 0:
        return 1.0
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

    sigma_hr_sq = signal.convolve(hr_padded ** 2, gauss, mode='valid') - mu_hr ** 2
    sigma_sr_sq = signal.convolve(sr_padded ** 2, gauss, mode='valid') - mu_sr ** 2
    sigma_hr_sr = signal.convolve(hr_padded * sr_padded, gauss, mode='valid') - mu_hr * mu_sr

    ssim_map = ((2 * mu_hr * mu_sr + c1) * (2 * sigma_hr_sr + c2)) / \
               ((mu_hr**2 + mu_sr**2 + c1) * (sigma_hr_sq + sigma_sr_sq + c2))
    return ssim_map.mean()

def r2_score_1d(hr, sr):
    ss_res = np.sum((hr - sr) ** 2)
    ss_tot = np.sum((hr - np.mean(hr)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-10))

# ===============================
# Evaluation (solo inferenza)
# ===============================
def evaluate_model(model, dataloader):
    model.eval()
    mse_crit = nn.MSELoss()
    mses, rmses, psnrs, ssims, r2s, inf_times = [], [], [], [], [], []

    print("Valutazione in corso...")
    with torch.no_grad():
        for lr_input, hr_target in tqdm(dataloader, desc="Test", unit="seg", leave=False):
            lr_input = lr_input.to(device)
            hr_target = hr_target.to(device)

            start = time.time()
            _, sr_recon = model(lr_input)
            inf_time = time.time() - start
            inf_times.append(inf_time)

            mse = mse_crit(sr_recon, hr_target).item()
            mses.append(mse)
            rmses.append(np.sqrt(mse))

            sr = sr_recon.cpu().numpy()[0]
            hr = hr_target.cpu().numpy()[0]

            psnr_vals = [psnr1d(hr[ch], sr[ch]) for ch in range(num_channels)]
            ssim_vals = [ssim1d(hr[ch], sr[ch]) for ch in range(num_channels)]
            r2_vals = [r2_score_1d(hr[ch], sr[ch]) for ch in range(num_channels)]

            psnrs.append(np.mean(psnr_vals))
            ssims.append(np.mean(ssim_vals))
            r2s.append(np.mean(r2_vals))

    return {
        'MSE': np.mean(mses),
        'RMSE': np.mean(rmses),
        'PSNR': np.mean(psnrs),
        'SSIM': np.mean(ssims),
        'R²': np.mean(r2s),
        'Avg Inf. Time (s)': np.mean(inf_times),
        'Segments': len(dataloader),
        'Parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
# ===============================
# Main: carica modello + valuta
# ===============================
if __name__ == "__main__":

    # --- Carica modello ---
    print(f"Caricamento modello da: {model_path}")
    model = util_sd.DCAE_SR_Base(num_channels=num_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Modello caricato.")

    # --- Loop su tutti i soggetti e tutti i run (assumendo 1-9 soggetti, 1-6 run per dataset standard EEG MI) ---
    results_list = []
    for subject in tqdm(range(1, 110), desc="Evaluating subjects", unit="subject"):
        for run in range(1, 14):
            # --- Test set (soggetto corrente, run corrente) ---
            test_dataset = util_sd.EEGDataset(subject_ids=[subject], runs=[run], project_path=project_path, add_noise=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            print(f"\nEvaluating subject {subject}, run {run}")
            print(f"Segmenti di test: {len(test_dataset)}")

            # --- Valutazione ---
            results = evaluate_model(model, test_loader)
            results['Subject'] = subject
            results['Run'] = run
            results_list.append(results)

    # --- Crea DataFrame con tutti i risultati ---
    df = DataFrame(results_list)
    df = df[['Subject', 'Run', 'MSE', 'RMSE', 'PSNR', 'SSIM', 'R²', 'Avg Inf. Time (s)', 'Segments', 'Parameters']]
    df = df.round(6)

    # --- Stampa tabella ---
    print("\n" + "="*60)
    print("        RISULTATI VALUTAZIONE MODELLO SU TUTTI I SOGGETTI E RUN")
    print("="*60)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=".6f"))
    print("="*60)

    # --- Salva in CSV ---
    csv_path = os.path.join(project_path, "standard_evaluation_results_all_subjects_runs.csv")
    df.to_csv(csv_path, index=False)
    print(f"Risultati salvati in: {csv_path}")