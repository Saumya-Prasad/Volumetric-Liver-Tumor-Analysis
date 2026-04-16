import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss3D(nn.Module):
    def __init__(self, window_size=7, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        self.register_buffer('kernel', self._gaussian_kernel(window_size))

    def _gaussian_kernel(self, size):
        sigma = 1.5
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
        return kernel.unsqueeze(0).unsqueeze(0)  # (1,1,k,k,k)

    def forward(self, pred, target):
        kernel = self.kernel.to(pred.device)
        pad = self.window_size // 2

        mu1 = F.conv3d(pred, kernel, padding=pad)
        mu2 = F.conv3d(target, kernel, padding=pad)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(pred * pred, kernel, padding=pad) - mu1_sq
        sigma2_sq = F.conv3d(target * target, kernel, padding=pad) - mu2_sq
        sigma12 = F.conv3d(pred * target, kernel, padding=pad) - mu1_mu2

        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = numerator / (denominator + 1e-8)
        return 1.0 - ssim_map.mean()


class FrequencyLoss(nn.Module):
    def forward(self, pred, target):
        pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
        target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        return F.l1_loss(pred_mag, target_mag)


class EdgeLoss3D(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_1d = torch.tensor([-1.0, 0.0, 1.0])
        smooth_1d = torch.tensor([1.0, 2.0, 1.0]) / 4.0

        def make_3d_kernel(direction):
            if direction == 'x':
                k = torch.einsum('i,j,k->ijk', smooth_1d, smooth_1d, sobel_1d)
            elif direction == 'y':
                k = torch.einsum('i,j,k->ijk', smooth_1d, sobel_1d, smooth_1d)
            else:
                k = torch.einsum('i,j,k->ijk', sobel_1d, smooth_1d, smooth_1d)
            return k.unsqueeze(0).unsqueeze(0)

        self.register_buffer('kx', make_3d_kernel('x'))
        self.register_buffer('ky', make_3d_kernel('y'))
        self.register_buffer('kz', make_3d_kernel('z'))

    def gradient_magnitude(self, x):
        kx = self.kx.to(x.device)
        ky = self.ky.to(x.device)
        kz = self.kz.to(x.device)
        gx = F.conv3d(x, kx, padding=1)
        gy = F.conv3d(x, ky, padding=1)
        gz = F.conv3d(x, kz, padding=1)
        return torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)

    def forward(self, pred, target):
        edge_pred = self.gradient_magnitude(pred)
        edge_target = self.gradient_magnitude(target)
        return F.l1_loss(edge_pred, edge_target)


class CombinedLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.1, w_freq=0.01, w_edge=0.1):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_freq = w_freq
        self.w_edge = w_edge
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss3D()
        self.freq = FrequencyLoss()
        self.edge = EdgeLoss3D()

    def forward(self, pred, target):
        l1_val = self.l1(pred, target)
        ssim_val = self.ssim(pred, target)
        freq_val = self.freq(pred, target)
        edge_val = self.edge(pred, target)

        total = (self.w_l1 * l1_val +
                 self.w_ssim * ssim_val +
                 self.w_freq * freq_val +
                 self.w_edge * edge_val)

        return total, {
            'l1': l1_val.item(),
            'ssim': ssim_val.item(),
            'freq': freq_val.item(),
            'edge': edge_val.item(),
            'total': total.item()
        }
