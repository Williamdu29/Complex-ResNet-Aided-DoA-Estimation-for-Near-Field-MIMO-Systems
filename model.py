# train_model.py
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------- hyperparams ----------
data_dir = "data"
batch_size = 128
epochs = 20
lr = 1e-3
Nin = 33   # 与生成数据时的Nin保持一致
# ----------------------------------------

# ---------- helper complex modules ----------
class ComplexConv1d(nn.Module):
    """
    Complex 1D conv implemented with two real Conv1d:
      out_re = conv(Wr, x_re) - conv(Wi, x_im)
      out_im = conv(Wi, x_re) + conv(Wr, x_im)
    Input: tuple (x_re, x_im) with shapes [B, C=2, L]
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, bias=True):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2  # SAME padding
        self.conv_re = nn.Conv1d(in_ch, out_ch, kernel_size,
                                 stride=stride, padding=padding, bias=bias)
        self.conv_im = nn.Conv1d(in_ch, out_ch, kernel_size,
                                 stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        # x: tuple (x_re, x_im)
        x_re, x_im = x
        a = self.conv_re(x_re) - self.conv_im(x_im)
        b = self.conv_im(x_re) + self.conv_re(x_im)
        return a, b # out_re, out_im

class ComplexLinear(nn.Module): # 支持复数输入与复数权重，输出仍然是复数
    """Complex fully-connected: handle complex vector (re,im) -> (re,im)"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # represent complex weight by two real matrices
        self.Wr = nn.Parameter(torch.randn(out_features, in_features) * math.sqrt(2/in_features)) # 初始化 kaiming
        self.Wi = nn.Parameter(torch.randn(out_features, in_features) * math.sqrt(2/in_features)) # 初始化 kaiming
        if bias:
            self.br = nn.Parameter(torch.zeros(out_features))
            self.bi = nn.Parameter(torch.zeros(out_features))
        else:
            self.br = None
            self.bi = None

    def forward(self, x):
        # x: tuple (re, im) each [B, in_features]
        xr, xi = x
        # out_re = Wr xr - Wi xi + br
        out_re = xr.matmul(self.Wr.t()) - xi.matmul(self.Wi.t())
        out_im = xr.matmul(self.Wi.t()) + xi.matmul(self.Wr.t())
        if self.br is not None:
            out_re = out_re + self.br
            out_im = out_im + self.bi
        return out_re, out_im

# activations for complex: Ctanh and Csigmoid
def ctanh(x):
    xr, xi = x
    return torch.tanh(xr), torch.tanh(xi)

def csigmoid(x):
    xr, xi = x
    return torch.sigmoid(xr), torch.sigmoid(xi)

# phase mapping: map complex (re,im) -> real phase rho in radians

# we first ensure re>0 by using csigmoid prior to this call
def phase_mapping(x):
    """
    rho = atan( (c - c*) / (c + c*) )
    where c = xr + j*xi.
    """
    xr, xi = x                        # xr, xi shape = [B, D]

    # build complex tensor c = xr + j xi
    c = torch.complex(xr, xi)         # complex tensor [B, D]
    numerator = c - torch.conj(c)     # c - c*
    denominator = c + torch.conj(c)   # c + c*
    # compute ratio
    ratio = numerator / denominator   # complex division
    imag_ratio = ratio.imag           # this equals (v / u)
    # compute arctan 
    rho = torch.atan(imag_ratio)      # shape [B, D]

    return rho

class ComplexResidualBlock(nn.Module):
    """Two complex conv layers with Ctanh nonlinearity and a complex shortcut.
       Configurable in/out channels.
    """
    def __init__(self, in_ch, mid_ch, out_ch, kernel=3):
        super().__init__()
        # first conv: in_ch -> mid_ch
        self.conv1 = ComplexConv1d(in_ch, mid_ch, kernel_size=kernel)
        # second conv: mid_ch -> out_ch
        self.conv2 = ComplexConv1d(mid_ch, out_ch, kernel_size=kernel)
        # if in/out channels differ, use 1x1 complex conv for shortcut
        if in_ch != out_ch:
            self.shortcut = ComplexConv1d(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.shortcut = None
        # activation is Ctanh after conv
    def forward(self, x):
        # x is tuple (re, im)
        xr, xi = x
        out = self.conv1((xr, xi))
        out = ctanh(out)
        out = self.conv2(out)
        out = ctanh(out)
        if self.shortcut is not None:
            sc = self.shortcut((xr, xi))
        else:
            sc = (xr, xi)
        # elementwise add
        out_re = out[0] + sc[0]
        out_im = out[1] + sc[1]
        return out_re, out_im

# ---------- build the ComplexResNet per table ----------
class ComplexResNet(nn.Module):
    def __init__(self, Nin):
        super().__init__()

        self.expected_Nin = Nin

        # Residual Block 1: (3×1×8) + (3×8×8)
        self.res1 = ComplexResidualBlock(1, 8, 8, kernel=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Residual Block 2: (3×8×4) + (3×4×4)
        self.res2 = ComplexResidualBlock(8, 4, 4, kernel=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- compute Nflat ---
        dummy_re = torch.zeros(1, 1, Nin)
        dummy_im = torch.zeros(1, 1, Nin)
        with torch.no_grad():
            r, i = self.forward_conv((dummy_re, dummy_im))
            C, L = r.shape[1], r.shape[2]
            self.Nflat = C * L

        # complex affine → 20
        self.comp_affine = ComplexLinear(self.Nflat, 20)

        # real affine layers: 20 → 10 → 10 → 1
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

        self.tanh = nn.Tanh()

    def forward_conv(self, x):
        xr, xi = x
        out = self.res1((xr, xi))
        out = (self.pool1(out[0]), self.pool1(out[1]))

        out = self.res2(out)
        out = (self.pool2(out[0]), self.pool2(out[1]))
        return out

    def forward(self, x):
        xr = x[:, 0:1, :] # real part
        xi = x[:, 1:2, :] # imaginary part

        out_re, out_im = self.forward_conv((xr, xi))

        B = out_re.shape[0]
        cr = out_re.view(B, -1)
        ci = out_im.view(B, -1)

        # complex → 20
        ar, ai = self.comp_affine((cr, ci))

        # Csigmoid → constrain real>0
        ar_s = torch.sigmoid(ar)
        ai_s = torch.sigmoid(ai)

        # phase mapping to real rho
        rho = phase_mapping((ar_s, ai_s))

        # real FC 20 → 10 → 10 → 1
        x = self.tanh(self.fc1(rho))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze(-1)
    

if __name__ == "__main__":   


    # ---------------- load data ----------------
    def load_data(data_dir):
        X_train = torch.load(os.path.join(data_dir, "X_train.pt"))
        y_train = torch.load(os.path.join(data_dir, "y_train.pt"))
        X_test = torch.load(os.path.join(data_dir, "X_test.pt"))
        y_test = torch.load(os.path.join(data_dir, "y_test.pt"))

        # [N, 1, 2, Nin] -> [N, 2, Nin]
        if X_train.dim() == 4 and X_train.size(1) == 1:
            X_train = X_train.squeeze(1)
            X_test = X_test.squeeze(1)

        # X shape expected [N, 2, Nin]; y shape [N, 2] where second dim is [theta, r]
        # We only predict angle; take theta in degrees -> convert to radians (paper uses radians inside)
        # Paper trains on angle in radians within (-pi/2, pi/2); we will scale to radians.
        # y_train[:,0] is theta degrees -> convert to radians
        y_train_theta = torch.tensor(y_train[:,0], dtype=torch.float32) * math.pi / 180.0
        y_test_theta = torch.tensor(y_test[:,0], dtype=torch.float32) * math.pi / 180.0
        return X_train, y_train_theta, X_test, y_test_theta

    X_train, y_train_theta, X_test, y_test_theta = load_data(data_dir)
    print("Loaded dataset shapes:", X_train.shape, y_train_theta.shape, X_test.shape, y_test_theta.shape)

    # create dataloaders
    train_dataset = TensorDataset(X_train, y_train_theta)
    test_dataset = TensorDataset(X_test, y_test_theta)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ---------------- model, loss, optimizer ----------------
    model = ComplexResNet(Nin).to(device)
    # use MAE loss (paper says MAE performed better for small angle errors)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    best_val = 1e9
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            optimizer.zero_grad()
            preds = model(xb)  # preds in radians
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
        val_loss = val_loss / len(test_loader.dataset)

        print(f"Epoch {epoch}/{epochs}  Train MAE (rad): {train_loss:.6f}   Val MAE (rad): {val_loss:.6f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_complex_resnet.pth")
            print("  -> saved best model")

    print("Training finished. Best val MAE (rad):", best_val)
