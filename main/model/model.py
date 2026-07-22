import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde
from torchdiffeq import odeint
import math
import time
from main.model.interpolation import (KernelInterpolation, WeightedKernelInterpolation, 
                                      GPInterpolation, WeightedGPInterpolation)
from main.model.mamba.mamba_ssm import Mamba, MambaConfig

class BaseModel(nn.Module):
    def __init__(self, t_grid=None):
        super().__init__()
        self.fit_time_accum = 0.0
        if t_grid is not None:
            self.register_buffer('t_grid', t_grid)
        else:
            self.t_grid = None

    def reset_fit_timer(self):
        self.fit_time_accum = 0.0

    def _time_interp(self, coeffs, interp_fn):
        if coeffs.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        X = interp_fn()
        
        if coeffs.is_cuda:
            torch.cuda.synchronize()
        self.fit_time_accum += (time.perf_counter() - t0)
        return X
    
    def inspect_interpolation(self, coeffs):
            self.eval()
            with torch.no_grad():
                X = self.make_interpolation(coeffs)
                if X is None:
                    return None
                    
                t0, t1 = X.interval[0], X.interval[1]
                
                eps = (t1 - t0) * 0.05  
                t_start = t0 + eps
                t_mid = (t0 + t1) / 2
                t_end = t1 - eps
                
                deriv_start = X.derivative(t_start)[0]
                deriv_mid = X.derivative(t_mid)[0]
                deriv_end = X.derivative(t_end)[0]
                
            return (t_start, t_mid, t_end), (deriv_start, deriv_mid, deriv_end)

class CDEStatsMixin:
    def __init__(self):
        self.nfe = 0
        self.last_tanh_saturation = 0.0
        self.last_tanh_mean = 0.0

    def _update_stats(self, z2):
        with torch.no_grad():
            abs_z2 = z2.abs()
            self.last_tanh_mean = abs_z2.mean().item()
            self.last_tanh_saturation = (abs_z2 > 0.95).float().mean().item()

class SimpleCDEFunc(nn.Module, CDEStatsMixin):
    def __init__(self, input_channels, hidden_channels, seq_len):
        super().__init__()
        CDEStatsMixin.__init__(self)
        self.input_channels, self.hidden_channels = input_channels, hidden_channels
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_channels * input_channels),
            nn.Tanh()
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                bound = 1 / math.sqrt(m.in_features)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, t, z):
        self.nfe += 1
        res = self.net(z)
        self._update_stats(res)
        return res.view(z.size(0), self.hidden_channels, self.input_channels)

class ParallelCDEFunc(nn.Module, CDEStatsMixin):
    def __init__(self, input_channels, hidden_channels, num_heads, seq_len):
        super().__init__()
        CDEStatsMixin.__init__(self)
        self.input_channels, self.hidden_channels, self.num_heads = input_channels, hidden_channels, num_heads
        
        self.w1 = nn.Parameter(torch.empty(num_heads, hidden_channels, hidden_channels))
        self.b1 = nn.Parameter(torch.empty(num_heads, hidden_channels))
        self.w2 = nn.Parameter(torch.empty(num_heads, hidden_channels * input_channels, hidden_channels))
        self.b2 = nn.Parameter(torch.empty(num_heads, hidden_channels * input_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        for w in [self.w1, self.w2]: 
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.hidden_channels)
        for b in [self.b1, self.b2]: 
            nn.init.uniform_(b, -bound, bound)

    def forward(self, t, z):
        self.nfe += 1
        batch_size = z.size(0) // self.num_heads
        z_in = z.view(batch_size, self.num_heads, self.hidden_channels)
        
        h = F.relu(torch.einsum('hoi,bhi->bho', self.w1, z_in) + self.b1)
        out = torch.tanh(torch.einsum('hoi,bhi->bho', self.w2, h) + self.b2)
        
        self._update_stats(out)
        return out.reshape(z.size(0), self.hidden_channels, self.input_channels)

class NeuralCDE(BaseModel):
    def __init__(self, input_channels, hidden_channels, output_channels, seq_len, interpolation="cubic", kernel_params={}, tol=1e-4, add_time=True, t_grid=None):
        super().__init__(t_grid)
        self.func = SimpleCDEFunc(input_channels, hidden_channels, seq_len)
        self.initial = nn.Linear(input_channels, hidden_channels)
        self.readout = nn.Linear(hidden_channels, output_channels)
        self.params = {'interp': interpolation, 'kp': kernel_params, 'tol': tol, 'at': add_time}
        
        # --- LEARNABLE ПАРАМЕТРЫ ---
        self.learnable_interp = kernel_params.get('learnable', False)
        if self.learnable_interp:
            if interpolation == 'kernel':
                self.bw_param = nn.Parameter(torch.tensor(kernel_params.get('bandwidth', 1.0), dtype=torch.float32))
            elif interpolation == 'gp':
                self.ls_param = nn.Parameter(torch.tensor(kernel_params.get('length_scale', 1.0), dtype=torch.float32))
                self.noise_param = nn.Parameter(torch.tensor(kernel_params.get('noise_std', 0.01), dtype=torch.float32))

    def make_interpolation(self, coeffs):
        it, kp_dict, at = self.params['interp'], dict(self.params['kp']), self.params['at']
        t = self.t_grid if self.t_grid is not None else (coeffs[0,:,0] if at else None)
        
        if self.learnable_interp:
            if it == 'kernel':
                kp_dict['bandwidth'] = F.softplus(self.bw_param) + 1e-4
            elif it == 'gp':
                kp_dict['length_scale'] = F.softplus(self.ls_param) + 1e-4
                kp_dict['noise_std'] = F.softplus(self.noise_param) + 1e-5
        
        if it in ['cubic', 'smoothing_spline']: return torchcde.CubicSpline(coeffs, t=t)
        if it == 'linear': return torchcde.LinearInterpolation(coeffs, t=t)
        if it == 'kernel': return KernelInterpolation(coeffs, t, kp_dict, at)
        if it == 'gp': return GPInterpolation(coeffs, t, kp_dict, at)
        return None

    def forward(self, coeffs):
        self.func.nfe = 0
        X = self._time_interp(coeffs, lambda: self.make_interpolation(coeffs))
        
        z0 = self.initial(X.evaluate(X.interval[0]))
        z_T = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval, atol=self.params['tol'], rtol=self.params['tol'])
        return self.readout(z_T[:, 1])

class MultiHeadCDEBase(BaseModel):
    def __init__(self, input_channels, hidden_channels, output_channels, seq_len, params, add_time, t_grid):
        super().__init__(t_grid)
        p = params
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.kernel = p.get('kernel', 'gaussian')
        self.bandwidths = p.get('bandwidths', [1.0])
        self.noise_std = p.get('noise_std', 0.01)
        self.tol = p.get('tol', 1e-4)
        self.aggregation = p.get('aggregation', 'concat')
        self.add_time = add_time
        
        self.num_heads = len(self.bandwidths)
        
        # --- LEARNABLE ПАРАМЕТРЫ ДЛЯ МУЛЬТИГОЛОВ ---
        self.learnable_interp = p.get('learnable', False)
        if self.learnable_interp:
            self.bandwidths_param = nn.Parameter(torch.tensor(self.bandwidths, dtype=torch.float32))
            if self.kernel == 'gp':
                self.noise_std_param = nn.Parameter(torch.tensor(self.noise_std, dtype=torch.float32))
        
        self.cde_func = ParallelCDEFunc(input_channels, hidden_channels, self.num_heads, seq_len)
        self.initial_layer = nn.Linear(input_channels, hidden_channels)
        r_dim = hidden_channels * self.num_heads if self.aggregation == 'concat' else hidden_channels
        self.readout = nn.Linear(r_dim, output_channels)

    def _get_weighted_interp(self, coeffs, weights):
        batch_size = coeffs.size(0)
        c_par = coeffs.repeat_interleave(self.num_heads, dim=0)
        w_par = weights.reshape(-1, weights.size(-1))
        
        if self.learnable_interp:
            b_val = F.softplus(self.bandwidths_param) + 1e-4
            n_val = F.softplus(self.noise_std_param) + 1e-5 if self.kernel == 'gp' else None
        else:
            b_val = torch.tensor(self.bandwidths, device=coeffs.device, dtype=coeffs.dtype)
            n_val = self.noise_std
            
        b_tensor = b_val.repeat(batch_size)
        if self.kernel == 'gp':
            return WeightedGPInterpolation(c_par, w_par, self.t_grid, {'length_scale': b_tensor, 'noise_std': n_val}, self.add_time)
        return WeightedKernelInterpolation(c_par, w_par, self.t_grid, {'kernel': self.kernel, 'bandwidth': b_tensor}, self.add_time)

    def _aggregate(self, z_T, batch_size):
        final_flat = z_T[:, 1, :]
        final = final_flat.view(batch_size, self.num_heads, self.hidden_channels)
        if self.aggregation == 'concat': return final.reshape(batch_size, -1)
        if self.aggregation == 'mean': return final.mean(dim=1)
        if self.aggregation == 'max': return final.max(dim=1)[0]
        return final.reshape(batch_size, -1)

class QFormerCDE(MultiHeadCDEBase):
    def __init__(self, input_channels, hidden_channels, output_channels, seq_len, qformer_params={}, add_time=True, t_grid=None):
        super().__init__(input_channels, hidden_channels, output_channels, seq_len, qformer_params, add_time, t_grid)
        raw_dim = input_channels
        self._last_attn = None 
        self.queries = nn.Parameter(torch.randn(self.num_heads, raw_dim))

    def forward(self, coeffs):
        self.cde_func.nfe = 0
        def make_it():
            safe_coeffs = torch.nan_to_num(coeffs, nan=0.0)
            attn_scores = torch.einsum('md,bld->bml', self.queries, safe_coeffs) / math.sqrt(self.queries.size(-1))
            start_feat = 1 if self.add_time else 0
            missing_mask = torch.isnan(coeffs[..., start_feat]).unsqueeze(1) 
            attn_scores = attn_scores.masked_fill(missing_mask, -1e9)
            attn = F.softmax(attn_scores, dim=-1)
            self._last_attn = attn 
            return self._get_weighted_interp(coeffs, attn) 
        
        X = self._time_interp(coeffs, make_it) 
        z0 = self.initial_layer(X.evaluate(X.interval[0]))
        z_T = torchcde.cdeint(X=X, func=self.cde_func, z0=z0, t=X.interval, atol=self.tol, rtol=self.tol)
        return self.readout(self._aggregate(z_T, coeffs.size(0)))

class ConvCDE(MultiHeadCDEBase):
    def __init__(self, input_channels, hidden_channels, output_channels, seq_len, conv_params={}, add_time=True, t_grid=None):
        super().__init__(input_channels, hidden_channels, output_channels, seq_len, conv_params, add_time, t_grid)
        raw_dim = input_channels
        self._last_attn = None 
        ks = conv_params.get('conv_kernel_size', 3)
        self.net = nn.Sequential(
            nn.Conv1d(raw_dim, hidden_channels, ks, padding=ks//2, padding_mode='replicate'), 
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, ks, padding=ks//2, padding_mode='replicate'), 
            nn.ReLU()
        )
        self.to_heads = nn.Linear(hidden_channels, self.num_heads)

    def forward(self, coeffs):
        self.cde_func.nfe = 0
        def make_it():
            safe_coeffs = torch.nan_to_num(coeffs, nan=0.0)
            x = self.net(safe_coeffs.permute(0, 2, 1)).permute(0, 2, 1)
            attn_scores = self.to_heads(x).transpose(1, 2)
            start_feat = 1 if self.add_time else 0
            missing_mask = torch.isnan(coeffs[..., start_feat]).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(missing_mask, -1e9)
            attn = F.softmax(attn_scores, dim=-1)
            self._last_attn = attn
            return self._get_weighted_interp(coeffs, attn)
            
        X = self._time_interp(coeffs, make_it) 
        z0 = self.initial_layer(X.evaluate(X.interval[0]))
        z_T = torchcde.cdeint(X=X, func=self.cde_func, z0=z0, t=X.interval, atol=self.tol, rtol=self.tol,)
        return self.readout(self._aggregate(z_T, coeffs.size(0)))

class ODERNNFunc(nn.Module, CDEStatsMixin):
    def __init__(self, hidden_channels):
        super().__init__()
        CDEStatsMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), 
            nn.Tanh(), 
            nn.Linear(hidden_channels, hidden_channels)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, h):
        self.nfe += 1
        return self.net(h)

class ODERNN(BaseModel):
    def __init__(self, input_channels, hidden_channels, output_channels, seq_len=None, tol=1e-3, add_time=True, t_grid=None, use_mask=True):
        super().__init__(t_grid)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.tol = tol
        self.add_time = add_time
        self.use_mask = use_mask
        
        self.feature_dim = input_channels - 1 if add_time else input_channels
        gru_input_dim = self.feature_dim * 2 if use_mask else self.feature_dim
        
        self.gru_cell = nn.GRUCell(gru_input_dim, hidden_channels)
        self.func = ODERNNFunc(hidden_channels)
        self.readout = nn.Linear(hidden_channels, output_channels)

    def forward(self, coeffs):
        batch_size, seq_len = coeffs.shape[:2]
        self.func.nfe = 0
        
        if self.add_time:
            times = coeffs[..., 0]      
            X = coeffs[..., 1:]         
        else:
            X = coeffs
            if self.t_grid is not None:
                times = self.t_grid.expand(batch_size, -1)
            else:
                times = torch.arange(seq_len, device=coeffs.device, dtype=coeffs.dtype).expand(batch_size, -1)

        mask = (~torch.isnan(X)).float()
        X_filled = torch.nan_to_num(X, nan=0.0)
        
        h = torch.zeros(batch_size, self.hidden_channels, device=coeffs.device)
        
        for i in range(seq_len):
            if i > 0:
                t0 = times[0, i-1]
                t1 = times[0, i]
                dt = t1 - t0
                if dt > 1e-5:
                    t_span = torch.stack([t0, t1])
                    out = odeint(self.func, h, t_span, atol=self.tol, rtol=self.tol)
                    h = out[1]

            x_i = X_filled[:, i, :] 
            if self.use_mask:
                m_i = mask[:, i, :]
                gru_input = torch.cat([x_i, m_i], dim=1)
                h_new = self.gru_cell(gru_input, h)
                obs_exists = (m_i.sum(dim=1, keepdim=True) > 0.5) 
                h = torch.where(obs_exists, h_new, h)
            else:
                h = self.gru_cell(x_i, h)

        return self.readout(h)

class GRUD(BaseModel):
    def __init__(self, input_channels, hidden_channels, output_channels, seq_len=None, add_time=True, t_grid=None, use_mask=True):
        super().__init__(t_grid)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.add_time = add_time
        self.use_mask = use_mask
        
        self.feature_dim = input_channels - 1 if add_time else input_channels
        gru_input_dim = self.feature_dim * 2 if use_mask else self.feature_dim
        
        self.gru_cell = nn.GRUCell(gru_input_dim, hidden_channels)
        self.decay_layer = nn.Linear(1, hidden_channels)
        self.readout = nn.Linear(hidden_channels, output_channels)
        
        nn.init.zeros_(self.decay_layer.weight)
        nn.init.zeros_(self.decay_layer.bias)

    def forward(self, coeffs):
        batch_size, seq_len = coeffs.shape[:2]
        if self.add_time:
            times = coeffs[..., 0]
            X = coeffs[..., 1:]
        else:
            X = coeffs
            if self.t_grid is not None:
                times = self.t_grid.expand(batch_size, -1)
            else:
                times = torch.arange(seq_len, device=coeffs.device, dtype=coeffs.dtype).expand(batch_size, -1)

        mask = (~torch.isnan(X)).float()
        X_filled = torch.nan_to_num(X, nan=0.0)
        
        h = torch.zeros(batch_size, self.hidden_channels, device=coeffs.device)
        
        for i in range(seq_len):
            if i > 0:
                dt = (times[:, i] - times[:, i-1]).unsqueeze(-1)
                gamma = torch.exp(-F.relu(self.decay_layer(dt)))
                h = h * gamma
            
            x_i = X_filled[:, i, :]
            if self.use_mask:
                m_i = mask[:, i, :]
                gru_input = torch.cat([x_i, m_i], dim=1)
                h_new = self.gru_cell(gru_input, h)
                obs_exists = (m_i.sum(dim=1, keepdim=True) > 0.5)
                h = torch.where(obs_exists, h_new, h)
            else:
                h = self.gru_cell(x_i, h)
            
        return self.readout(h)


class MambaModel(BaseModel):
    def __init__(self, input_channels, hidden_channels, output_channels, seq_len=None, 
                 add_time=True, t_grid=None, use_mask=True, mamba_params=None):
        super().__init__(t_grid)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.add_time = add_time
        self.use_mask = use_mask
        
        # Получаем параметры из словаря или ставим дефолты
        mp = mamba_params or {}
        n_layers      = mp.get('n_layers', 2)
        d_state       = mp.get('d_state', 16)
        d_conv        = mp.get('d_conv', 4)
        expand_factor = mp.get('expand_factor', 2)
        use_pscan     = mp.get('pscan', True) # Добавляем это
        
        self.feature_dim = input_channels - 1 if add_time else input_channels
        
        # Вход: [Фичи, Маска, Delta T] или только [Фичи]
        mamba_input_dim = (self.feature_dim * 2 + 1) if use_mask else self.feature_dim
        
        # Проекция в размерность модели (d_model)
        self.input_proj = nn.Linear(mamba_input_dim, hidden_channels)
        
        # Настройка конфига mambapy
        config = MambaConfig(
            d_model=hidden_channels, 
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            pscan=use_pscan  # <--- Передаем сюда!
        )
        self.mamba = Mamba(config)
        
        self.readout = nn.Linear(hidden_channels, output_channels)

    def forward(self, coeffs):
        batch_size, seq_len = coeffs.shape[:2]
        
        if self.add_time:
            times = coeffs[..., 0]
            X = coeffs[..., 1:]
        else:
            X = coeffs
            if self.t_grid is not None:
                times = self.t_grid.expand(batch_size, -1)
            else:
                times = torch.arange(seq_len, device=coeffs.device, dtype=coeffs.dtype).expand(batch_size, -1)

        mask = (~torch.isnan(X)).float()
        X_filled = torch.nan_to_num(X, nan=0.0)
        
        if self.use_mask:
            dt = torch.zeros_like(times)
            dt[:, 1:] = times[:, 1:] - times[:, :-1]
            dt = dt.unsqueeze(-1)
            mamba_in = torch.cat([X_filled, mask, dt], dim=-1)
        else:
            mamba_in = X_filled
        
        h = self.input_proj(mamba_in)
        h = self.mamba(h)
        # Читаем по последнему шагу (B, L, D) -> (B, D)
        out = self.readout(h[:, -1, :])
        
        return out
    
def compute_logsig_windows(x, window_size, depth):
    B, L, C = x.shape
    
    num_windows = L // window_size
    if num_windows == 0:
        raise ValueError(f"Window size {window_size} is larger than sequence length {L}")
        
    x_trimmed = x[:, :num_windows*window_size, :]
    x_windows = x_trimmed.view(B, num_windows, window_size, C)
    
    x_start = x_windows[:, :, 0, :]
    x_end = x_windows[:, :, -1, :]
    
    displacement = x_end - x_start 
    
    if depth == 1:
        return displacement
        
    elif depth == 2:
        path_in_window = x_windows - x_start.unsqueeze(2)  
        
        dX = path_in_window[:, :, 1:, :] - path_in_window[:, :, :-1, :]
        
        X_mid = (path_in_window[:, :, 1:, :] + path_in_window[:, :, :-1, :]) / 2.0
        
        iterated_integral = torch.einsum('bnwi,bnwj->bnij', X_mid, dX)
        
        B_dim, N_dim, _, _ = iterated_integral.shape
        second_level = iterated_integral.view(B_dim, N_dim, -1)
        
        return torch.cat([displacement, second_level], dim=-1)
        
    else:
        raise ValueError("Only Depth 1 and 2 are supported for this implementation.")
    

class LogNeuralCDE(BaseModel):
    def __init__(self, input_channels, hidden_channels, output_channels, seq_len, 
                 log_ncde_params={}, tol=1e-4, add_time=True, t_grid=None):
        super().__init__(t_grid)
        
        self.step_size = int(log_ncde_params.get('step_size', 5))
        self.depth = int(log_ncde_params.get('depth', 1))
        self.hidden_channels = hidden_channels
        self.tol = tol
        self.add_time = add_time 
        
        self.raw_input_channels = input_channels 
        
        if self.depth == 1:
            self.cde_input_dim = self.raw_input_channels
        elif self.depth == 2:
            self.cde_input_dim = self.raw_input_channels + (self.raw_input_channels ** 2)
        
        self.func = SimpleCDEFunc(self.cde_input_dim, hidden_channels, seq_len // self.step_size)
        self.initial = nn.Linear(self.cde_input_dim, hidden_channels)
        self.readout = nn.Linear(hidden_channels, output_channels)
        
    def forward(self, coeffs):
        self.func.nfe = 0
        
        if coeffs.is_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        logsig_path = compute_logsig_windows(coeffs, self.step_size, self.depth)
        
        X = torchcde.LinearInterpolation(logsig_path)
        
        if coeffs.is_cuda: torch.cuda.synchronize()
        self.fit_time_accum += (time.perf_counter() - t0)
        
        z0 = self.initial(X.evaluate(X.interval[0]))
        
        z_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.interval, atol=self.tol, rtol=self.tol)
        
        z_T_final = z_T[:, 1]
        return self.readout(z_T_final)
