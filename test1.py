import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import dcae_sr_eeg_motor_imagery_subpixel as util_pixel
from scipy import signal
import dcae_sr_eeg_motor_imagery as util_sd
from metrics_calculator import psnr1d, ssim1d, r2_score_1d

# -------------------------------
# Configurazioni
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Percorsi
project_path = r"C:\Users\RITA\OneDrive\Desktop\Tesi_Magistrale_Ammendola"
model_path = os.path.join(project_path, "dcae_sr_eeg_motor_imagery_subpixel_n1_031125.pth")  # <-- Cambia se hai un modello salvato

# Parametri
num_channels = 64
lr_window_length = 250
hr_window_length = 2500
batch_size = 1

# Scegli il modello da testare
ModelClass = util_pixel.DCAE_SR_SubPixel  # <-- Cambia con il modello che vuoi testare
model_name = "DCAE-SR con PixelShuffle e residuale"

# -------------------------------
# Carica un singolo sample dal dataset di test
# -------------------------------
from metrics_calculator import EEGDataset  # Importa il tuo dataset

# Usa solo 1 soggetto per il test
test_subjects = [1]  # Cambia se vuoi un altro soggetto
test_dataset = EEGDataset(subject_ids=test_subjects, runs=[1], project_path=project_path, add_noise=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Prendi il primo sample
lr_input, hr_target = next(iter(test_loader))
lr_input = lr_input.to(device)
hr_target = hr_target.to(device)

# -------------------------------
# Carica o inizializza il modello
# -------------------------------
model = ModelClass().to(device)

# Opzione 1: Carica modello pre-addestrato (se esiste)
if os.path.exists(model_path):
    print(f"Caricamento modello pre-addestrato da: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Modello non trovato. Usa modello non addestrato (solo per test struttura).")
    print("Per risultati reali, addestra e salva il modello prima.")

model.eval()

# -------------------------------
# Inferenza su singolo sample
# -------------------------------
with torch.no_grad():
    lr_recon, sr_recon = model(lr_input)
    sr_recon = sr_recon.cpu().numpy()[0]  # (64, 2500)
    lr_input_np = lr_input.cpu().numpy()[0]  # (64, 250)
    hr_target_np = hr_target.cpu().numpy()[0]  # (64, 2500)

# -------------------------------
# Visualizzazione: 3 canali di esempio
# -------------------------------
channels_to_plot = [0, 10, 30]  # Scegli 3 canali rappresentativi
time_lr = np.arange(lr_window_length)
time_hr = np.arange(hr_window_length)

plt.figure(figsize=(15, 10))

for i, ch in enumerate(channels_to_plot):
    plt.subplot(3, 1, i + 1)

    # Downsample HR per confronto visivo con LR (opzionale)
    hr_downsampled = signal.resample(hr_target_np[ch], lr_window_length)

    plt.plot(time_hr, hr_target_np[ch], label='HR Ground Truth', color='green', alpha=0.7)
    plt.plot(time_hr, sr_recon[ch], label='SR Reconstructed', color='blue', linewidth=1.5)

    # Mostra LR upscalato per confronto
    lr_upsampled = np.repeat(lr_input_np[ch], 10)  # 250 -> 2500 (x10)
    lr_upsampled = lr_upsampled[:hr_window_length]  # Taglia a 2500
    plt.plot(time_hr, lr_upsampled, label='LR (Nearest Upsample)', color='red', alpha=0.6, linestyle='--')

    plt.title(f'Canale {ch} - {model_name}')
    plt.xlabel('Campioni')
    plt.ylabel('Ampiezza (uV)')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle("Ricostruzione EEG - Confronto HR vs SR vs LR (upsampled)", fontsize=16, y=0.98)
plt.show()

# -------------------------------
# Calcolo metriche sul sample
# -------------------------------

psnr_vals = []
ssim_vals = []
r2_vals = []

for ch in range(num_channels):
    psnr_vals.append(psnr1d(hr_target_np[ch], sr_recon[ch]))
    ssim_vals.append(ssim1d(hr_target_np[ch], sr_recon[ch]))
    r2_vals.append(r2_score_1d(hr_target_np[ch], sr_recon[ch]))

print("\n" + "=" * 50)
print(f"METRICHE MEDIE SUL SINGOLO SAMPLE ({model_name})")
print("=" * 50)
print(f"PSNR:  {np.mean(psnr_vals):.2f} dB")
print(f"SSIM:  {np.mean(ssim_vals):.4f}")
print(f"R²:    {np.mean(r2_vals):.4f}")
print(f"Numero parametri modello: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print("=" * 50)