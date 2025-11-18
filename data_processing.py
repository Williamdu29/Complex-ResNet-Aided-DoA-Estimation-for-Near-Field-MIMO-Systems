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
    A = steering_vector_nearfield(thetas, rs, N, d, lam).unsqueeze(1)  # [N,1] (M=1)

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
        chi_l_t = math.floor(nc - t / 2)
        chi_r_t = math.floor(nc - (t - 1) / 2)

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

'''
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
'''

#   生成数据集
def generate_dataset(N, M, K, d, lam, SNR_dB, thetas, rs, Nin=33, verbose=True):
    X_list = []
    y_list = []

    total = len(thetas) * len(rs)
    count = 0

    for r in rs: # 200λ-1800λ
        for th in thetas: # -90°-90°
            Y = generate_nearfield_data(N, M, K, d, lam, th, r, SNR_dB) # 生成接收信号矩阵
            R = sample_covariance(Y) # 计算样本协方差矩阵
            Rv = reconstruct_vcm(R) # 重构虚拟协方差矩阵
            Rc = crop_matrix(Rv, Nin) # 中心裁剪
            Xi = extract_signal_subspace(Rc, M) # 提取信号子空间
            X = complex_to_tensor(Xi) # 转成实数tensor

            X_list.append(X.unsqueeze(0)) # 这是输入 shape=[1, 2, Nin]
            y_list.append([th, r])  # label = (theta, distance)

            # ======== 打印进度 ========
            count += 1

            if verbose:
                '''
                # 方案 A：逐条打印
                print(f"[{count}/{total}] θ={th:.2f}°, r={r/lam:.1f}λ 样本生成完成")
                '''

                # 方案 B：每 1000 条打印一次 
                if count % 1000 == 0:
                    print(f"[{count}/{total}] 生成中...")

    X_data = torch.cat(X_list, dim=0)   # [num, 2, Nin] batch of inputs
    y_data = torch.tensor(y_list, dtype=torch.float32)

    return X_data, y_data



# 训练集 / 测试集采样范围

theta_train = np.arange(-45, 45.01, 0.01)
theta_test  = np.arange(-45, 45.01, 0.1)
r_vals = np.arange(200, 1800.1, 25) * lam

# 生成训练集
thetas_sel = theta_train
rs_sel = r_vals

print("Generating training dataset...")
print(f"θ范围: {len(thetas_sel)} 个角度样本")
print(f"r范围: {len(rs_sel)} 个距离样本")
print(f"总样本数: {len(thetas_sel) * len(rs_sel)}")

X_train, y_train = generate_dataset(N, M, K, d, lam, 
                                    SNR_dB, thetas_sel, rs_sel,Nin=33, verbose=True)

print("Training dataset generated.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Example y_train (first 5):", y_train[:5])


# 生成测试集
print("Generating test dataset...")

theta_test_sel = theta_test   # [-45,45] 每 0.1°
rs_test_sel = rs_sel          # 与训练集一致

print(f"θ范围: {len(theta_test_sel)} 个角度样本")
print(f"r范围: {len(rs_test_sel)} 个距离样本")
print(f"总样本数: {len(theta_test_sel) * len(rs_test_sel)}")

X_test, y_test = generate_dataset(N, M, K, d, lam,
                                  SNR_dB, theta_test_sel, rs_test_sel, Nin=33, verbose=True)

print("Test dataset generated.")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)





import os

save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

torch.save(X_train, os.path.join(save_dir, "X_train.pt"))
torch.save(y_train, os.path.join(save_dir, "y_train.pt"))

print("✔ Training data saved:")
print(f"  - {os.path.join(save_dir, 'X_train.pt')}")
print(f"  - {os.path.join(save_dir, 'y_train.pt')}")

torch.save(X_test, os.path.join(save_dir, "X_test.pt"))
torch.save(y_test, os.path.join(save_dir, "y_test.pt"))

print("✔ Test data saved:")