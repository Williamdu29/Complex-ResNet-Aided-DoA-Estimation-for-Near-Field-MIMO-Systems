# run_simulations.py
import os
import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

# --- ADJUST PATHS HERE ---
MODEL_PATH = "best_complex_resnet.pth"   # trained model file (update if needed)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- IMPORT MODEL CLASS ---

from model import ComplexResNet 

# --- PARAMETERS ---
fc = 28e9
c = 3e8
lam = c / fc
N_full = 65
d = lam / 2
nc = (N_full - 1) / 2
M = 1
# Nin used when cropping VCM in training pipeline
Nin = 33
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Reuse data-generation functions (adapted) ---
def steering_vector_nearfield(theta_deg, r, N=N_full, d=d, lam=lam):
    theta_rad = math.radians(theta_deg)
    nc_local = (N - 1) / 2
    delta = torch.arange(N, dtype=torch.float32) - nc_local  # 使用当前 N 计算
    # ensure tensors on cpu float
    delta = delta.float()
    r_n = torch.sqrt(r**2 + (delta * d)**2 - 2 * r * delta * d * math.sin(theta_rad))
    phase = -2 * math.pi / lam * (r_n - r)
    a = (r / r_n) * torch.exp(1j * phase)
    return a / torch.sqrt(torch.tensor(N, dtype=torch.complex64))

def generate_nearfield_snapshots(theta_deg, r, N=N_full, M=1, K=100, d=d, lam=lam, SNR_dB=10):
    """
    Generate Y (N x K) noisy snapshots for single source (M=1)
    Returns complex tensor Y of shape [N, K] (complex64)
    """
    A = steering_vector_nearfield(theta_deg, r, N, d, lam).unsqueeze(1)  # [N,1]
    # complex Gaussian source (M x K)
    s = (torch.randn(M, K) + 1j * torch.randn(M, K)) / math.sqrt(2.0)
    y = A @ s  # [N, K]
    # noise
    Ps = torch.mean(torch.abs(y)**2)
    sigma2 = Ps / (10**(SNR_dB / 10.0))
    # generate real gaussian parts explicitly (use same device)
    real_noise = torch.randn(y.shape, dtype=torch.float32, device=y.device)
    imag_noise = torch.randn(y.shape, dtype=torch.float32, device=y.device)
    noise = torch.sqrt(torch.tensor(sigma2 / 2.0, dtype=torch.float32, device=y.device)) * (real_noise + 1j * imag_noise)
    y_noisy = y + noise
    return y_noisy

def sample_covariance(Y):
    R = (Y @ Y.conj().T) / Y.shape[1]
    R = (R + R.conj().T) / 2
    return R

def reconstruct_vcm(R):
    """
    Robust VCM reconstruction: works for any R.shape[0].
    Fills diagonals with averaged values; uses centered indices computed
    from current N.
    """
    N = R.shape[0]
    vcm = torch.zeros_like(R, dtype=R.dtype, device=R.device)
    nc_local = (N - 1) / 2.0

    for t in range(-N+1, N):
        if t == 0:
            diag_vals = torch.diagonal(R)
            mean_val = torch.mean(diag_vals)
            diag_block = torch.ones((N,), dtype=R.dtype, device=R.device) * mean_val
            vcm += torch.diag(diag_block)
            continue

        chi_l_t = math.floor(nc_local - t / 2.0)
        chi_r_t = math.floor(nc_local - (t - 1) / 2.0)
        chi_l_neg_t = int(round(nc_local + t / 2.0))
        chi_r_neg_t = int(round(nc_local + (t - 1) / 2.0))

        if (0 <= chi_l_t < N) and (0 <= chi_l_neg_t < N) and (0 <= chi_r_t < N) and (0 <= chi_r_neg_t < N):
            val = 0.5 * (R[chi_l_t, chi_l_neg_t] + R[chi_r_t, chi_r_neg_t])
        else:
            val = torch.tensor(0.0, dtype=R.dtype, device=R.device)

        diag_len = N - abs(t)
        diag_block = torch.ones((diag_len,), dtype=R.dtype, device=R.device) * val
        vcm += torch.diag(diag_block, diagonal=t)

    vcm = (vcm + vcm.conj().T) / 2
    return vcm


def crop_matrix(R, Nin_local):
    """
    中心裁剪到 Nin_local；若 Nin_local > N 则居中 zero-pad 到 Nin_local。
    R: (N x N) complex tensor
    返回 (Nin_local x Nin_local) tensor
    """
    N = R.shape[0]
    if Nin_local == N:
        return R
    if Nin_local < N:
        start = (N - Nin_local) // 2
        end = start + Nin_local
        return R[start:end, start:end]
    # Nin_local > N : zero-pad 居中
    pad = Nin_local - N
    left = pad // 2
    right = pad - left
    R_padded = torch.zeros((Nin_local, Nin_local), dtype=R.dtype, device=R.device)
    R_padded[left:left+N, left:left+N] = R
    return R_padded


def extract_signal_subspace(Rcrop, M=1):
    eigvals, eigvecs = torch.linalg.eigh(Rcrop)
    idx = torch.argsort(eigvals, descending=True)
    Xi_s = eigvecs[:, idx[:M]]  # [Nin, M]
    return Xi_s

def complex_to_tensor(Xc):
    # Xc: [Nin, M] complex, for M=1 returns [1, 2, Nin] (channels real/imag)
    X_real = Xc.real.T.unsqueeze(1)  # [M,1,Nin]
    X_imag = Xc.imag.T.unsqueeze(1)
    return torch.cat([X_real, X_imag], dim=1)  # [M,2,Nin]

def build_input_from_snapshots(Y, Nin=Nin):
    """
    Pipeline: Y (N x K) -> sample covariance -> reconstruct VCM -> crop -> subspace -> complex tensor shape [1,2,Nin]
    """
    R = sample_covariance(Y)
    Rv = reconstruct_vcm(R)
    Rc = crop_matrix(Rv, Nin)
    Xi = extract_signal_subspace(Rc, M=1)  # [Nin,1]
    Xc = complex_to_tensor(Xi)  # [1,2,Nin]
    
    return Xc # shape [1,2,Nin]

# --- Helper: angular error (wrap properly) ---
def angular_error_rad(pred, true):  # 角度是周期性的，直接相减可能不对
    # pred, true are scalars in radians
    diff = pred - true
    # wrap to [-pi, pi]
    diff = (diff + math.pi) % (2*math.pi) - math.pi # wrap difference to [-pi, pi]
    return diff

# --- Load model ---
print("Loading model...")
model = ComplexResNet(Nin).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
# 读取模型期望的输入长度（由 model.py 中设置）
model_expected_Nin = getattr(model, "expected_Nin", Nin)
print(f"Model expected Nin = {model_expected_Nin}")

# 将“构造网络输入时的裁剪尺寸”固定为模型期望值（论文的 cropped-VCM）
Nin_crop_used = model_expected_Nin

print("Model loaded and set to eval on", device)

# --- Monte-Carlo and utilities ---
@torch.no_grad()
def predict_theta_rad(model, input_tensor):
    # input_tensor shape: [1,2,Nin]
    t = input_tensor.to(device).float()
    out = model(t)  # model returns shape [B] with radians
    return out.cpu().item() # 模型的输出是 DoA（方向角）估计值，单位是弧度（rad）

def monte_carlo_rmse(model,
                     theta_deg,
                     r_lambda,
                     N_array,
                     SNR_dB,
                     K,
                     trials=200):
    """
    Monte-Carlo DOA estimation RMSE (in radians)
    for a given:
        - theta_deg（角度）
        - r_lambda（距离，单位 λ）
        - N_array（阵元数 N）
        - SNR_dB
        - snapshots K
    """

    theta_true_rad = math.radians(theta_deg)
    r = r_lambda * lam

    errors = []

    for _ in range(trials):

        #  Step 1: 生成近场快拍 
        Y = generate_nearfield_snapshots(
            theta_deg,
            r,
            N_array,   # 用不同阵元数 N
            M=1,
            K=K,
            d=d,
            lam=lam,
            SNR_dB=SNR_dB
        )

        #  Step 2: 构造模型输入（Nin 固定为 33）
        Xin = build_input_from_snapshots(Y, Nin_crop_used)  # [1, 2, 33]

        #  Step 3: 推理 
        pred = predict_theta_rad(model, Xin)

        #  Step 4: 计算角度误差 
        err = angular_error_rad(pred, theta_true_rad)
        errors.append(err)

    errors = np.array(errors)
    rmse = math.sqrt(np.mean(errors ** 2))
    return rmse, errors


# ---------------------------
# --- RMSE vs SNR (for several distances) ---
# ---------------------------
snr_list = np.arange(-10, 11, 2)  # -10 .. 10 dB step 2
distances_lambda = [600, 800, 1000, 1200]  # λ units 
K_fixed = 100
trials = 200
N_array = 65

results_snr = {}
print("Running RMSE vs SNR (this will take a while)...")
for r_lambda in distances_lambda:
    rmses = []
    for snr in snr_list:
        rmse, _ = monte_carlo_rmse(model, theta_deg=0.0, N_array=N_array, r_lambda=r_lambda, SNR_dB=snr, K=K_fixed, trials=trials)
        rmses.append(rmse)
        print(f"r={r_lambda}λ, SNR={snr} dB -> RMSE (rad) {rmse:.4f}")
    results_snr[r_lambda] = np.array(rmses)

plt.figure()
for r_lambda, rmses in results_snr.items():
    plt.plot(snr_list, rmses, marker='o', label=f"{r_lambda} λ")
plt.xlabel("SNR (dB)")
plt.ylabel("RMSE (rad)")
plt.title("RMSE vs SNR for different distances")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "rmse_vs_snr.png"), dpi=200)
print("Saved rmse_vs_snr.png")

# --- RMSE vs #antennas (use cropping idea) ---
# paper examines different numbers of antennas
Nin_list = [33, 65, 129, 257]  # test different cropped/zero-padded input sizes
SNR_list = np.arange(-10, 11, 2)  # -10 .. 10 dB step 2
r_lambda_test = 1000
trials = 200
K_fixed = 100

rmse_results = {}

print("Running RMSE vs SNR for multiple Nin...")

# 关键：模型期望的输入尺寸（在 model.py 中 self.expected_Nin = Nin）
model_expected_Nin = model.expected_Nin
print(f"[INFO] Model expected input Nin = {model_expected_Nin}")

for N_array_test in Nin_list:

    rmse_results[N_array_test] = []

    print(f"\n=== Testing array size N = {N_array_test} ===")

    for snr in SNR_list:

        rmse, _ = monte_carlo_rmse(
            model,
            theta_deg = 0.0,
            r_lambda  = r_lambda_test,
            N_array   = N_array_test,         # 只改变实际阵元数
            SNR_dB    = snr,
            K         = K_fixed,
            trials    = trials
        )

        rmse_results[N_array_test].append(rmse)
        print(f"SNR={snr} dB → RMSE={rmse:.4f}")

# --- Plotting ---
plt.figure(figsize=(7,5))
markers = ["o", ">", "s", "<"]
colors = ["blue", "orange", "green", "red"]

for i, N_array_test in enumerate(Nin_list):
    plt.plot(
        SNR_list,
        rmse_results[N_array_test],
        marker=markers[i],
        color=colors[i],
        label=f"N_array = {N_array_test}",
        linewidth=2
    )

plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.grid(True)
plt.legend()
plt.title("RMSE vs SNR for different array sizes", fontsize=14)
plt.savefig(os.path.join(RESULTS_DIR, "rmse_vs_SNR_multiN.png"), dpi=300)
print("Saved rmse_vs_SNR_multiN.png")


# ---------------------------
# --- RMSE vs SNR for different K ---
# ---------------------------

snr_list = np.arange(-10, 11, 2)     # SNR = -10 : 2 : 10
K_list = [16, 32, 64, 128]           # Different number of snapshots
theta_test = 0.0                     # DOA fixed at 0°
r_lambda_test = 1000                 # r = 1000 λ (paper typical)
trials = 200                         # Monte-Carlo trials

N_array = 65                      # Fixed array size

results_fig9a = {}

print("Running simulation ...")

for K_val in K_list:
    rmse_list = []
    for snr in snr_list:
        rmse, _ = monte_carlo_rmse(
            model,
            theta_deg=theta_test,
            r_lambda=r_lambda_test,
            N_array=N_array,
            SNR_dB=snr,
            K=K_val,
            trials=trials
        )
        rmse_list.append(rmse)
        print(f"K={K_val}, SNR={snr} dB → RMSE {rmse:.4f}")

    results_fig9a[K_val] = np.array(rmse_list)

# ---- Plot ----
plt.figure(figsize=(6,5))
markers = ['o', 's', 'v', '<']
for (K_val, m) in zip(K_list, markers):
    plt.plot(snr_list, results_fig9a[K_val], marker=m, label=f"K = {K_val}")

plt.xlabel("SNR (dB)")
plt.ylabel("RMSE")
plt.title("RMSE vs SNR for different K")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "RMSE_vs_SNR_multiK.png"), dpi=300)
print("Saved RMSE_vs_SNR_multiK.png")




# ---------------------------
# --- RMSE vs SNR for different Nin × K ---
# ---------------------------

snr_list = np.arange(-10, 11, 2)
Nin_candidates = [33, 65]
K_candidates = [64, 128]
theta_test = 0.0
r_lambda_test = 1000
trials = 200

results_fig9b = {}

print("Running simulation ...")

for Nin_test in Nin_candidates:
    for K_val in K_candidates:

        rmse_list = []
        for snr in snr_list:
            # 注意：这里 N_array 是实际阵列元个数；Nin_test 是我们想用来 crop/pad 的输入尺寸
            # 使用 monte_carlo_rmse 中已固定的 Nin_crop_used。若你希望用不同 Nin_crop 做实验，
            # 直接临时覆盖 Nin_crop_used_local 传给 build_input_from_snapshots（下方注释示例）
            rmse, _ = monte_carlo_rmse(
                model,
                theta_deg=theta_test,
                r_lambda=r_lambda_test,
                N_array=Nin_test,
                SNR_dB=snr,
                K=K_val,
                trials=trials
            )
            rmse_list.append(rmse)
            print(f"Nin={Nin_test}, K={K_val}, SNR={snr} → RMSE {rmse:.4f}")

        results_fig9b[(Nin_test, K_val)] = np.array(rmse_list)

# ---- Plot ----
plt.figure(figsize=(6,5))
markers = ['o', 'o--', 'v', 'v--']
for (key, marker) in zip(results_fig9b.keys(), markers):
    Nin_test, K_val = key
    label = f"Nin={Nin_test}, K={K_val}"
    plt.plot(snr_list, results_fig9b[key], marker=marker[0], linestyle=marker[1:], label=label)

plt.xlabel("SNR (dB)")
plt.ylabel("RMSE")
plt.title("RMSE vs SNR for different Nin × K")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "RMSE_vs_SNR_multiNxK.png"), dpi=300) # dpi=300 for better quality
print("Saved RMSE_vs_SNR_multiNxK.png")

