from scipy.stats import ttest_rel
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score, mean_squared_error
from dcae_sr_eeg_motor_imagery_subpixel import *
from dcae_sr_eeg_motor_imagery import *
from tqdm import tqdm  # <--- BARRA DI PROGRESSO


# ===============================
# Config
# ===============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_path = r"C:\Users\RITA\OneDrive\Desktop\Tesi_Magistrale_Ammendola"
model_path = os.path.join(project_path, "dcae_sr_eeg_motor_imagery_subpixel_n1_031125.pth")
num_channels = 64


# ===============================
# Funzioni per metriche
# ===============================
def compute_metrics(hr_true, sr_pred):
    hr_true = hr_true.flatten()
    sr_pred = sr_pred.flatten()

    mse = mean_squared_error(hr_true, sr_pred)
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(np.max(np.abs(hr_true)) / (np.sqrt(mse) + 1e-8))  # evita div 0
    r2 = r2_score(hr_true, sr_pred)
    data_range = hr_true.max() - hr_true.min()
    ssim_val = ssim(hr_true, sr_pred, data_range=data_range if data_range > 0 else 1.0)
    t_stat, p_value = ttest_rel(hr_true, sr_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "PSNR": psnr,
        "SSIM": ssim_val,
        "R²": r2,
        "p-value": p_value
    }


# ===============================
# Carica modello
# ===============================
print(f"Caricamento modello da: {model_path}")
model = DCAE_SR_SubPixel(num_channels=num_channels).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===============================
# Dataset di test
# ===============================
test_dataset = EEGDataset(subject_ids=[1], runs=[3], project_path=project_path, add_noise=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Numero di segmenti nel test: {len(test_dataset)}")

# ===============================
# Testing con barra di progresso
# ===============================
all_metrics = []
first_plot = True

print("Inizio valutazione...")

for i, (lr_input, hr_target) in enumerate(tqdm(test_loader, desc="Test Progress", unit="seg")):
    lr_input, hr_target = lr_input.to(device), hr_target.to(device)

    with torch.no_grad():
        _, sr_output = model(lr_input)

    hr_true = hr_target.cpu().numpy()[0]  # [64, 2500]
    sr_pred = sr_output.cpu().numpy()[0]  # [64, 2500]

    # Metriche sul canale 0
    metrics = compute_metrics(hr_true[0], sr_pred[0])
    all_metrics.append(metrics)

    # --- Stampa progressiva ---
    print(f"\nSegmento {i + 1}/{len(test_loader)}")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v:.6f}")

    # --- Plot solo del primo ---
    if first_plot:
        plt.figure(figsize=(12, 5))
        plt.plot(hr_true[0], label='HR Ground Truth', alpha=0.8)
        plt.plot(sr_pred[0], label='SR Reconstructed', alpha=0.7)
        plt.title(f'EEG SR Reconstruction - Segmento {i + 1} (Canale 0)')
        plt.xlabel('Campioni (500 Hz)')
        plt.ylabel('Ampiezza (z-score)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        first_plot = False

# ===============================
# Media finale
# ===============================
if all_metrics:
    mean_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    std_metrics = {k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}

    print("\n" + "=" * 50)
    print("RISULTATI MEDI SUL TEST SET")
    print("=" * 50)
    for k in mean_metrics:
        print(f"{k:8s}: {mean_metrics[k]:.6f} ± {std_metrics[k]:.6f}")
    print("=" * 50)
else:
    print("Nessun dato valutato.")