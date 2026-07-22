import torch
import torchcde
import math

def _k_gauss(u, b): return (1 / (b * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * (u / b)**2)
def _dk_gauss(u, b): return -u / (b**2) * _k_gauss(u, b)
def _k_laplace(u, b): return (0.5 / b) * torch.exp(-torch.abs(u / b))
def _dk_laplace(u, b): return -0.5 * torch.sign(u) / (b**2) * torch.exp(-torch.abs(u / b))
def _k_cauchy(u, b): return 1 / (math.pi * b * (1 + (u / b)**2))
def _dk_cauchy(u, b): return -2 * u / (math.pi * (b**2) * b * (1 + (u / b)**2)**2)
def _k_epan(u, b): return (0.75 / b) * torch.clamp(1 - (u / b)**2, min=0)
def _dk_epan(u, b): return (-1.5 * u / (b**3)) * (torch.abs(u / b) < 1).float()
def _k_tri(u, b): return (70 / (81 * b)) * torch.clamp(1 - torch.abs(u / b)**3, min=0)**3
def _dk_tri(u, b): 
    ua = torch.abs(u / b)
    return -(210 / (81 * b)) * (ua**2 * torch.sign(u) / b) * torch.clamp(1 - ua**3, min=0)**2 * (ua < 1).float()

KERNELS = {
    'gaussian': (_k_gauss, _dk_gauss), 'laplacian': (_k_laplace, _dk_laplace),
    'cauchy': (_k_cauchy, _dk_cauchy), 'epanechnikov': (_k_epan, _dk_epan), 'tricube': (_k_tri, _dk_tri)
}

class _BaseInterp(torchcde.interpolation_base.InterpolationBase):
    def __init__(self, coeffs, t, include_time, **kwargs):
        super().__init__(**kwargs)
        self.include_time = include_time
        if t is not None:
            self.register_buffer('_t', t) 
        else:
            self.register_buffer('_t', coeffs[0, :, 0])
            
        if self.include_time:
            self.register_buffer('_x', coeffs[..., 1:])
            self.register_buffer('_true_t', coeffs[..., 0]) 
        else:
            self.register_buffer('_x', coeffs)

    @property
    def grid_points(self): return self._t
    @property
    def interval(self): return torch.stack([self._t[0], self._t[-1]])

    def _pad_time(self, t_eval, res, is_deriv=False):
        if not self.include_time: 
            return res
        
        s_diff = self._t[-1] - self._t[0]
        t_diff = self._true_t[..., -1] - self._true_t[..., 0]
        dt_ds = t_diff / s_diff  
        
        if is_deriv:
            t_chan = dt_ds.unsqueeze(-1) 
        else:
            if t_eval.dim() == 0:
                s_val = t_eval.expand(self._true_t.size(0))
            else:
                s_val = t_eval
            t_start = self._true_t[..., 0]
            t_chan = (t_start + (s_val - self._t[0]) * dt_ds).unsqueeze(-1)
                
        return torch.cat([t_chan, res], dim=-1)
    
class KernelInterpolation(_BaseInterp):
    def __init__(self, coeffs, t=None, kernel_params={}, include_time=True, **kwargs):
        super().__init__(coeffs, t, include_time, **kwargs)
        self.bandwidth = kernel_params.get('bandwidth', 0.1)
        self._k_fn, self._dk_fn = KERNELS[kernel_params.get('kernel', 'gaussian')]

    def _eval_core(self, t_eval, get_deriv=False):
        t_eval_b = t_eval.expand(self._x.size(0)) if t_eval.dim() == 0 else t_eval
        diff = t_eval_b.unsqueeze(-1) - self._t
        
        missing_mask = torch.isnan(self._x[..., 0])
        safe_x = torch.nan_to_num(self._x, nan=0.0)
        
        K = self._k_fn(diff, self.bandwidth)
        K = K * (~missing_mask).float() 
        
        D = K.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        N = (K.unsqueeze(-1) * safe_x).sum(dim=1)
        
        if not get_deriv: return self._pad_time(t_eval, N / D)
        
        dK = self._dk_fn(diff, self.bandwidth)
        dK = dK * (~missing_mask).float() 
        
        dD, dN = dK.sum(dim=-1, keepdim=True), (dK.unsqueeze(-1) * safe_x).sum(dim=1)
        return self._pad_time(t_eval, (dN * D - N * dD) / D.pow(2).clamp(min=1e-10), True)

    def evaluate(self, t): return self._eval_core(t, False)
    def derivative(self, t): return self._eval_core(t, True)

class WeightedKernelInterpolation(_BaseInterp):
    def __init__(self, coeffs, weights, t=None, kernel_params={}, include_time=True, **kwargs):
        super().__init__(coeffs, t, include_time, **kwargs)
        self.weights = weights.unsqueeze(-1)
        bw = kernel_params.get('bandwidth', 0.1)
        self.bandwidth = bw if torch.is_tensor(bw) else torch.tensor(bw, device=coeffs.device)
        # ИСПРАВЛЕНИЕ: Безопасное приведение к 3D для градиентов
        if self.bandwidth.dim() == 1: self.bandwidth = self.bandwidth.view(-1, 1, 1)
        self._k_fn, self._dk_fn = KERNELS[kernel_params.get('kernel', 'gaussian')]

    def _eval_core(self, t_eval, get_deriv=False):
        t_eval_b = t_eval.expand(self._x.size(0)) if t_eval.dim() == 0 else t_eval
        diff = t_eval_b.unsqueeze(-1) - self._t
        
        missing_mask = torch.isnan(self._x[..., 0])
        safe_x = torch.nan_to_num(self._x, nan=0.0)
        
        K = (self._k_fn(diff, self.bandwidth) * self.weights.squeeze(-1))
        K = K * (~missing_mask).float() 
        
        D = K.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        N = (K.unsqueeze(-1) * safe_x).sum(dim=1)
        
        if not get_deriv: return self._pad_time(t_eval, N / D)
        
        dK = self._dk_fn(diff, self.bandwidth) * self.weights.squeeze(-1)
        dK = dK * (~missing_mask).float() 
        
        dD, dN = dK.sum(dim=-1, keepdim=True), (dK.unsqueeze(-1) * safe_x).sum(dim=1)
        return self._pad_time(t_eval, (dN * D - N * dD) / D.pow(2).clamp(min=1e-20), True)

    def evaluate(self, t): return self._eval_core(t, False)
    def derivative(self, t): return self._eval_core(t, True)

class GPInterpolation(_BaseInterp):
    def __init__(self, coeffs, t=None, gp_params={}, include_time=True, **kwargs):
        super().__init__(coeffs, t, include_time, **kwargs)
        self.length_scale, self.noise_std = gp_params.get('length_scale', 1.0), gp_params.get('noise_std', 1e-2)
        
        K_tt = torch.exp(-0.5 * (self._t.unsqueeze(1) - self._t.unsqueeze(0)).pow(2) / self.length_scale**2)
        K_tt += (self.noise_std**2) * torch.eye(len(self._t), device=self._t.device)
        
        missing_mask = torch.isnan(self._x[..., 0])
        safe_x = torch.nan_to_num(self._x, nan=0.0)
        
        K_tt_batched = K_tt.unsqueeze(0).expand(self._x.size(0), -1, -1).clone()
        huge_noise = torch.diag_embed(missing_mask.float() * 1e6)
        K_tt_batched = K_tt_batched + huge_noise
        
        self.alpha = torch.linalg.solve(K_tt_batched, safe_x)

    def _eval_core(self, t_eval, get_deriv=False):
        t_eval_b = t_eval.expand(self._x.size(0)) if t_eval.dim() == 0 else t_eval
        diff = t_eval_b.unsqueeze(1) - self._t.unsqueeze(0)
        K_star = torch.exp(-0.5 * (diff / self.length_scale)**2)
        if get_deriv: K_star = K_star * (-diff / (self.length_scale**2))
        mu = torch.matmul(K_star.unsqueeze(1), self.alpha).squeeze(1)
        return self._pad_time(t_eval, mu, get_deriv)

    def evaluate(self, t): return self._eval_core(t, False)
    def derivative(self, t): return self._eval_core(t, True)

class WeightedGPInterpolation(_BaseInterp):
    def __init__(self, coeffs, weights, t=None, gp_params={}, include_time=True, **kwargs):
        super().__init__(coeffs, t, include_time, **kwargs)
        self.length_scale = gp_params.get('length_scale', 1.0)
        # ИСПРАВЛЕНИЕ: Безопасное приведение к 3D
        if torch.is_tensor(self.length_scale): 
            if self.length_scale.dim() == 1:
                self.length_scale = self.length_scale.view(-1, 1, 1)
        self.noise_std = gp_params.get('noise_std', 1e-2)
        
        diff_sq = (self._t.unsqueeze(1) - self._t.unsqueeze(0)).pow(2)
        K_tt = torch.exp(-0.5 * diff_sq.unsqueeze(0) / (self.length_scale**2))
        noise_mat = torch.diag_embed((self.noise_std**2) / (weights + 1e-5))
        
        # --- NaN Masking ---
        missing_mask = torch.isnan(self._x[..., 0])
        safe_x = torch.nan_to_num(self._x, nan=0.0)
        huge_noise = torch.diag_embed(missing_mask.float() * 1e6)
        
        self.alpha = torch.linalg.solve(K_tt + noise_mat + huge_noise, safe_x)

    def _eval_core(self, t_eval, get_deriv=False):
        t_eval_b = t_eval.expand(self._x.size(0)) if t_eval.dim() == 0 else t_eval
        diff = t_eval_b.view(-1, 1, 1) - self._t.view(1, 1, -1)
        K_star = torch.exp(-0.5 * diff.pow(2) / (self.length_scale**2))
        if get_deriv: K_star = K_star * (-diff / (self.length_scale**2))
        mu = torch.matmul(K_star, self.alpha).squeeze(1)
        return self._pad_time(t_eval, mu, get_deriv)

    def evaluate(self, t): return self._eval_core(t, False)
    def derivative(self, t): return self._eval_core(t, True)