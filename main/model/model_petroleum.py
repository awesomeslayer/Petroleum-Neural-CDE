import torch
import torch.nn.functional as F
import torchcde
from main.model.model import NeuralCDE, ODERNN, GRUD, QFormerCDE, ConvCDE
from torchdiffeq import odeint

class EmbeddingNeuralCDE(NeuralCDE):
    def forward(self, coeffs):
        self.func.nfe = 0
        X = self._time_interp(coeffs, lambda: self.make_interpolation(coeffs))
        z0 = self.initial(X.evaluate(X.interval[0]))
        t = self.t_grid if self.t_grid is not None else coeffs[0, :, 0]
        z_T = torchcde.cdeint(X=X, func=self.func, z0=z0, t=t, atol=self.params['tol'], rtol=self.params['tol'])
        
        # Получаем эмбеддинг и ОБЯЗАТЕЛЬНО L2-нормализуем его
        emb = self.readout(z_T[:, -1])
        return F.normalize(emb, p=2, dim=-1)

class EmbeddingQFormerCDE(QFormerCDE):
    def _aggregate_heads(self, z_T, batch_size):
        final = z_T.view(batch_size, self.num_heads, self.hidden_channels)
        if self.aggregation == 'concat': return final.reshape(batch_size, -1)
        if self.aggregation == 'mean': return final.mean(dim=1)
        if self.aggregation == 'max': return final.max(dim=1)[0]
        return final.reshape(batch_size, -1)

    def forward(self, coeffs):
        self.cde_func.nfe = 0
        def make_it():
            safe_coeffs = torch.nan_to_num(coeffs, nan=0.0)
            import math
            attn_scores = torch.einsum('md,bld->bml', self.queries, safe_coeffs) / math.sqrt(self.queries.size(-1))
            start_feat = 1 if self.add_time else 0
            missing_mask = torch.isnan(coeffs[..., start_feat]).unsqueeze(1) 
            attn_scores = attn_scores.masked_fill(missing_mask, -1e9)
            attn = F.softmax(attn_scores, dim=-1)
            self._last_attn = attn 
            return self._get_weighted_interp(coeffs, attn) 
        
        X = self._time_interp(coeffs, make_it) 
        z0 = self.initial_layer(X.evaluate(X.interval[0]))
        t = self.t_grid if self.t_grid is not None else coeffs[0, :, 0]
        z_T = torchcde.cdeint(X=X, func=self.cde_func, z0=z0, t=t, atol=self.tol, rtol=self.tol)
        
        emb = self.readout(self._aggregate_heads(z_T[:, -1], coeffs.size(0)))
        return F.normalize(emb, p=2, dim=-1)

class EmbeddingConvCDE(ConvCDE):
    def _aggregate_heads(self, z_T, batch_size):
        final = z_T.view(batch_size, self.num_heads, self.hidden_channels)
        if self.aggregation == 'concat': return final.reshape(batch_size, -1)
        if self.aggregation == 'mean': return final.mean(dim=1)
        if self.aggregation == 'max': return final.max(dim=1)[0]
        return final.reshape(batch_size, -1)

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
        t = self.t_grid if self.t_grid is not None else coeffs[0, :, 0]
        z_T = torchcde.cdeint(X=X, func=self.cde_func, z0=z0, t=t, atol=self.tol, rtol=self.tol)
        
        emb = self.readout(self._aggregate_heads(z_T[:, -1], coeffs.size(0)))
        return F.normalize(emb, p=2, dim=-1)

class EmbeddingODERNN(ODERNN):
    def forward(self, coeffs):
        batch_size, seq_len = coeffs.shape[:2]
        self.func.nfe = 0
        times = coeffs[..., 0] if self.add_time else self.t_grid.expand(batch_size, -1)
        X = coeffs[..., 1:] if self.add_time else coeffs
        mask = (~torch.isnan(X)).float()
        X_filled = torch.nan_to_num(X, nan=0.0)
        
        h = torch.zeros(batch_size, self.hidden_channels, device=coeffs.device)
        for i in range(seq_len):
            if i > 0:
                t0, t1 = times[0, i-1], times[0, i]
                if (t1 - t0) > 1e-5:
                    out = odeint(self.func, h, torch.stack([t0, t1]), atol=self.tol, rtol=self.tol)
                    h = out[1]
            x_i = X_filled[:, i, :] 
            if self.use_mask:
                m_i = mask[:, i, :]
                h_new = self.gru_cell(torch.cat([x_i, m_i], dim=1), h)
                h = torch.where((m_i.sum(dim=1, keepdim=True) > 0.5), h_new, h)
            else:
                h = self.gru_cell(x_i, h)
                
        emb = self.readout(h)
        return F.normalize(emb, p=2, dim=-1)

class EmbeddingGRUD(GRUD):
    def forward(self, coeffs):
        batch_size, seq_len = coeffs.shape[:2]
        times = coeffs[..., 0] if self.add_time else self.t_grid.expand(batch_size, -1)
        X = coeffs[..., 1:] if self.add_time else coeffs
        mask = (~torch.isnan(X)).float()
        X_filled = torch.nan_to_num(X, nan=0.0)
        
        h = torch.zeros(batch_size, self.hidden_channels, device=coeffs.device)
        for i in range(seq_len):
            if i > 0:
                dt = (times[:, i] - times[:, i-1]).unsqueeze(-1)
                h = h * torch.exp(-F.relu(self.decay_layer(dt)))
            x_i = X_filled[:, i, :]
            if self.use_mask:
                m_i = mask[:, i, :]
                h_new = self.gru_cell(torch.cat([x_i, m_i], dim=1), h)
                h = torch.where(m_i.sum(dim=1, keepdim=True) > 0.5, h_new, h)
            else:
                h = self.gru_cell(x_i, h)
                
        emb = self.readout(h)
        return F.normalize(emb, p=2, dim=-1)