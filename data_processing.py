import torch
import numpy as np
import math

#  基本参数 
fc = 28e9                  # 载波频率
c = 3e8                    # 光速
lam = c / fc               # 波长
N = 65                     # 阵元数
d = lam / 2                # 阵元间距
nc = (N - 1) / 2           # 中心索引
M = 1                      # 信号源数
K = 100                    # 快照数
SNR_dB = 10                # 信噪比(dB)
theta_deg = [10]           # 到达角（度）参看table 3
r_vals = [200 * lam]       # 源距离 论文中提到的Fresnel区间，200λ是典型的近场距离 （既非平面波，也不是太近） 参看table 3

#  函数定义 

def steering_vector_nearfield(theta, r, N, d, lam):
    """近场阵列流形矢量"""
    delta = torch.arange(N, dtype=torch.float32) - nc
    theta_rad = math.radians(theta) # 转为弧度
    r_n = torch.sqrt(r**2 + (delta * d)**2 - 2 * r * delta * d * torch.sin(torch.tensor(theta_rad))) # distance from the n-th array element to the m-th source
    phase = -2 * math.pi / lam * (r_n - r)
    a = (r / r_n) * torch.exp(1j * phase)
    return a / torch.sqrt(torch.tensor(N, dtype=torch.complex64))  # 归一化

def generate_nearfield_data(N, M, K, d, lam, thetas, rs, SNR_dB):
    """生成接收信号矩阵 Y"""
    A = torch.stack([steering_vector_nearfield(thetas[m], rs[m], N, d, lam) for m in range(M)], dim=1) # 阵列流形矩阵 [N, M]
    s = (torch.randn(M, K) + 1j * torch.randn(M, K)) / math.sqrt(2)  # 这里使用随机信号作为源信号矩阵 [M, K]
    y = A @ s # 接收信号矩阵 [N, K]
    # 添加噪声
    # 根据信噪比计算噪声功率
    Ps = torch.mean(torch.abs(y)**2) # 信号功率
    sigma2 = Ps / (10**(SNR_dB / 10)) # 噪声功率
    noise = torch.sqrt(sigma2/2) * (torch.randn_like(y) + 1j * torch.randn_like(y)) # 复高斯白噪声
    # randn_like 生成与 y 形状相同的实高斯噪声
    y_noisy = y + noise
    return y_noisy

def sample_covariance(Y):
    """样本协方差矩阵"""
    R = (Y @ Y.conj().T) / Y.shape[1]
    R = (R + R.conj().T) / 2  # 保证 Hermitian
    return R

def reconstruct_vcm(R):
    """虚拟协方差矩阵 VCM：按每条对角线取平均形成 Toeplitz"""
    N = R.shape[0]
    vcm = torch.zeros_like(R, dtype=R.dtype)
    for t in range(-N+1, N): # 遍历th对角线
        if t == 0:
            # 主对角线直接保留原值（或取平均）
            diag_vals = torch.diagonal(R) # 主对角线元素
            mean_val = torch.mean(diag_vals) # 取平均值
            vcm += torch.diag(torch.full((N,), mean_val, dtype=R.dtype)) # 主对角线赋值
            continue
        # 计算 χl(t), χr(t)
        chi_l_t = int(round(nc - t / 2))
        chi_r_t = int(round(nc - (t - 1) / 2))

        # 对称位置的索引 χl(-t), χr(-t)
        chi_l_neg_t = int(round(nc + t / 2))
        chi_r_neg_t = int(round(nc + (t - 1) / 2))

        # 检查索引是否在有效范围内
        if (0 <= chi_l_t < N) and (0 <= chi_l_neg_t < N) and \
           (0 <= chi_r_t < N) and (0 <= chi_r_neg_t < N):
            val = 0.5 * (R[chi_l_t, chi_l_neg_t] + R[chi_r_t, chi_r_neg_t]) # 取平均
        else:
            # 超出阵列范围，跳过或设为 0
            val = torch.tensor(0.0, dtype=R.dtype, device=R.device)

        # 用该平均值填充整条偏移为 t 的对角线
        diag_len = N - abs(t)
        vcm += torch.diag(torch.full((diag_len,), val, dtype=R.dtype), diagonal=t)

    # 保证 Hermitian
    vcm = (vcm + vcm.conj().T) / 2
    return vcm

def crop_matrix(R, Nin):
    """中心裁剪 Nin×Nin"""
    N = R.shape[0]
    start = (N - Nin) // 2
    end = start + Nin
    return R[start:end, start:end]

def extract_signal_subspace(Rcrop, M):
    """特征分解提取信号子空间"""
    eigvals, eigvecs = torch.linalg.eigh(Rcrop) # 得到特征值和特征向量
    idx = torch.argsort(eigvals, descending=True) # 降序排序索引
    Xi_s = eigvecs[:, idx[:M]] # 取前M个特征向量构成信号子空间
    return Xi_s

def complex_to_tensor(Xc):
    """将复向量转成2通道实数 tensor [M, 2, Nin]"""
    X_real = Xc.real.T.unsqueeze(1)
    X_imag = Xc.imag.T.unsqueeze(1)
    return torch.cat([X_real, X_imag], dim=1)

#  主流程 

Y = generate_nearfield_data(N, M, K, d, lam, theta_deg, r_vals, SNR_dB)
R = sample_covariance(Y)
R_vcm = reconstruct_vcm(R)
Nin = 33
R_crop = crop_matrix(R_vcm, Nin)
Xi_s = extract_signal_subspace(R_crop, M)
X_input = complex_to_tensor(Xi_s)

print("Input tensor shape for CVNN:", X_input.shape)
print("Example real part (first few):", X_input[0,0,:5])
print("Example imag part (first few):", X_input[0,1,:5])
