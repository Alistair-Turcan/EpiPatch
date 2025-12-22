import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from copy import deepcopy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from .utils import *
from .metrics import get_loss

US_POP_2019 = {
    "AL": 4903185, "AK": 731545,  "AZ": 7278717, "AR": 3017804, "CA": 39512223,
    "CO": 5758736, "CT": 3565287, "DE": 973764,  "DC": 705749,  "FL": 21477737,
    "GA": 10617423,"HI": 1415872, "ID": 1787065, "IL": 12671821,"IN": 6732219,
    "IA": 3155070, "KS": 2913314, "KY": 4467673, "LA": 4648794, "ME": 1344212,
    "MD": 6045680, "MA": 6892503, "MI": 9986857, "MN": 5639632, "MS": 2976149,
    "MO": 6137428, "MT": 1068778, "NE": 1934408, "NV": 3080156, "NH": 1359711,
    "NJ": 8882190, "NM": 2096829, "NY": 19453561,"NC": 10488084,"ND": 762062,
    "OH": 11689100,"OK": 3956971, "OR": 4217737, "PA": 12801989,"RI": 1059361,
    "SC": 5148714, "SD": 884659,  "TN": 6829174, "TX": 28995881,"UT": 3205958,
    "VT": 623989,  "VA": 8535519, "WA": 7614893, "WV": 1792147, "WI": 5822434,
    "WY": 578759,  "PR": 3193694, "NYC": 8804190
}


def resolve_population_2019(
    M,
    population=None,   # None | scalar | tensor/list length M
    regions=None,      # None | "CA" | list[str] length M
    device=None,
    dtype=None,
):
    """
    Returns N_vec: [M] tensor (population per node).

    Priority:
      1) explicit `population` (scalar or length-M)
      2) `regions` lookup in US_POP_2019 (scalar or length-M)
      3) fallback: ones (treat as normalized population)
    """
    device = device or "cpu"
    dtype = dtype or torch.float32

    # 1) explicit population provided
    if population is not None:
        if torch.is_tensor(population):
            pop = population.to(device=device, dtype=dtype)
            if pop.numel() == 1:
                return pop.view(1).expand(M)
            if pop.numel() == M:
                return pop.view(M)
            raise ValueError(f"population tensor must have numel 1 or {M}, got {pop.numel()}")
        # python number
        if isinstance(population, (int, float)):
            return torch.full((M,), float(population), device=device, dtype=dtype)
        # list/tuple/np array
        pop = torch.as_tensor(population, device=device, dtype=dtype).flatten()
        if pop.numel() == 1:
            return pop.expand(M)
        if pop.numel() == M:
            return pop
        raise ValueError(f"population must be scalar or length {M}, got {pop.numel()}")

    # 2) regions lookup
    if regions is not None:
        if isinstance(regions, str):
            if regions not in US_POP_2019:
                raise KeyError(f"Unknown region code: {regions}")
            return torch.full((M,), float(US_POP_2019[regions]), device=device, dtype=dtype)

        # list of region codes per node
        if isinstance(regions, (list, tuple)):
            if len(regions) != M:
                raise ValueError(f"regions must have length {M}, got {len(regions)}")
            vals = []
            for r in regions:
                if r not in US_POP_2019:
                    raise KeyError(f"Unknown region code: {r}")
                vals.append(float(US_POP_2019[r]))
            return torch.tensor(vals, device=device, dtype=dtype)

        raise ValueError("regions must be a string or list/tuple of strings")

    # 3) fallback
    return torch.ones((M,), device=device, dtype=dtype)

def _inv_tanh_0_1(y, eps=1e-6, device=None, dtype=torch.float32):
    """
    y in (0,1) -> x in R such that (tanh(x)+1)/2 = y
    """
    y = torch.as_tensor(y, device=device, dtype=dtype)
    y = torch.clamp(y, eps, 1 - eps)
    return torch.atanh(2.0 * y - 1.0)

class SIRm_tanh(nn.Module):
    """
    SIR with beta,gamma in (0,1) via tanh reparam.
    State: (S, I, R) in counts (or any consistent unit).
    """
    def __init__(self, population, parameter, dtype=torch.float32):
        super().__init__()
        self.register_buffer("N", torch.as_tensor(population, dtype=dtype))
        self.init_params(parameter, dtype=dtype)

    def init_params(self, params, dtype=torch.float32):
        device = self.N.device
        self.logbeta  = Parameter(_inv_tanh_0_1(params["beta"],  device=device, dtype=dtype), requires_grad=True)
        self.loggamma = Parameter(_inv_tanh_0_1(params["gamma"], device=device, dtype=dtype), requires_grad=True)

    def get_scaled_params(self, convert_cpu=False):
        beta  = (torch.tanh(self.logbeta) + 1.0) * 0.5
        gamma = (torch.tanh(self.loggamma) + 1.0) * 0.5
        out = {"beta": beta, "gamma": gamma}
        if convert_cpu:
            for k, v in out.items():
                out[k] = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu()
        return out

    def ODE(self, state, t=None):
        """
        state: [..., 3] last dim = (S,I,R)
        returns dstate with same shape
        """
        p = self.get_scaled_params()
        beta, gamma = p["beta"], p["gamma"]

        S = state[..., 0]
        I = state[..., 1]
        R = state[..., 2]

        N = self.N
        while N.dim() < S.dim():
            N = N.unsqueeze(0)

        new_inf_rate = beta * S * I / (N + 1e-12)   # flow S->I per unit time
        dS = -new_inf_rate
        dI = new_inf_rate - gamma * I
        dR = gamma * I
        return torch.stack([dS, dI, dR], dim=-1)

    def incidence(self, state):
        """
        Returns the incidence flow (new infections per unit time): beta*S*I/N
        state: [...,3]
        """
        p = self.get_scaled_params()
        beta = p["beta"]
        S = state[..., 0]
        I = state[..., 1]

        N = self.N
        while N.dim() < S.dim():
            N = N.unsqueeze(0)

        return beta * S * I / (N + 1e-12)

class SIRIncidenceRollout(nn.Module):
    """
    Produces a horizon-length incidence series from a latent SIR rollout.

    Inputs:
      feature: [B, W, M, F]
      Uses feature[:, -1, :, target_idx] as the latest observed incidence (cases/step).

    Output:
      cases_pred: [B, H, M]  (new cases per step)
    """
    def __init__(
        self,
        sir: SIRm_tanh,
        target_idx=0,
        dt=1.0,
        learn_r0_frac=False,
        r0_init=0.0,
        enforce_mass=True,
        obs="incidence",              # "incidence" or "ili_percent"
        outpatient_ratio=None,        # needed if obs == "ili_percent"
    ):
        super().__init__()
        self.sir = sir
        self.target_idx = int(target_idx)
        self.dt = float(dt)
        self.enforce_mass = bool(enforce_mass)
        self.obs = str(obs)
        self.outpatient_ratio = outpatient_ratio

        if learn_r0_frac:
            self.log_r0_frac = Parameter(_inv_tanh_0_1(r0_init, device=self.sir.N.device, dtype=self.sir.N.dtype))
        else:
            self.log_r0_frac = None

    def _infer_H(self, pred, target):
        x = pred if pred is not None else target
        if x is None or x.dim() != 3:
            raise ValueError("Need pred or target with shape [B,H,M] or [B,M,H] to infer H.")
        return x.shape[1] if x.shape[1] != x.shape[2] else x.shape[-1]

    def forward(self, feature, pred=None, target=None):
        if feature is None or feature.dim() != 4:
            raise ValueError("feature must be [B,W,M,F]")
        B, W, M, Fdim = feature.shape
        H = self._infer_H(pred, target)

        device = feature.device
        dtype = feature.dtype

        # --- Latest observed incidence (cases/step)
        C0 = feature[:, -1, :, self.target_idx].to(device=device, dtype=dtype)  # [B,M]
        C0 = torch.clamp(C0, min=0.0)

        # --- N as [B,M]
        N = self.sir.N.to(device=device, dtype=dtype)
        if N.numel() == 1:
            N_bm = N.view(1, 1).expand(B, M)
        elif N.dim() == 1 and N.numel() == M:
            N_bm = N.view(1, M).expand(B, M)
        else:
            N_bm = N.expand(B, M)

        # --- Infer initial I0 from incidence: C0 ≈ beta*S0*I0/N, with S0≈N => I0≈C0/beta
        beta = self.sir.get_scaled_params()["beta"].to(device=device, dtype=dtype)
        beta_safe = torch.clamp(beta, min=1e-4)

        I0 = torch.clamp(C0 / beta_safe, min=0.0, max=N_bm)

        # --- Optional R0 fraction
        if self.log_r0_frac is None:
            R0 = torch.zeros_like(I0)
        else:
            r0_frac = (torch.tanh(self.log_r0_frac) + 1.0) * 0.5  # scalar in (0,1)
            R0 = r0_frac * torch.clamp(N_bm - I0, min=0.0)

        S0 = torch.clamp(N_bm - I0 - R0, min=0.0)

        state = torch.stack([S0, I0, R0], dim=-1)  # [B,M,3]

        cases = []
        for _ in range(H):
            # incidence per unit time
            inc = self.sir.incidence(state)  # [B,M]
            # convert to "cases per step" (discrete) via dt
            c_step = self.dt * inc

            # map to observed if needed (EINN-style ILI% proxy)
            if self.obs == "ili_percent":
                if self.outpatient_ratio is None:
                    raise ValueError("outpatient_ratio must be provided for obs='ili_percent'")
                OR = float(self.outpatient_ratio)
                # EINN connects ILI% to incidence scaled by outpatient ratio (see paper).
                # Here we follow their idea: ILI% ≈ (beta*S*I/N) / (N*OR)
                c_step = inc / (N_bm * OR + 1e-12)  # fraction (not counts)
            elif self.obs != "incidence":
                raise ValueError(f"Unknown obs type: {self.obs}")

            cases.append(c_step)

            # Euler update
            dstate = self.sir.ODE(state)
            state = torch.clamp(state + self.dt * dstate, min=0.0)

            if self.enforce_mass:
                total = state.sum(dim=-1, keepdim=True)  # [B,M,1]
                state = state * (N_bm.unsqueeze(-1) / (total + 1e-12))

        return torch.stack(cases, dim=1)  # [B,H,M]

def compute_epi_ngm_forecast(
    adj_prob,          # [B,H,M,M]
    beta,              # [B,H,M]
    gamma,             # [B,H,M]
    x_last,            # [B,M]
    adj_static=None,   # [M,M] or [B,M,M]
    clamp_max=1.0,
    eps=1e-6
):
    assert adj_prob.dim() == 4, "adj_prob must be [B,H,M,M]"
    assert beta.dim() == 3 and gamma.dim() == 3, "beta/gamma must be [B,H,M]"
    assert x_last.dim() == 2, "x_last must be [B,M]"

    B, H, M, _ = adj_prob.shape
    device = adj_prob.device
    dtype = adj_prob.dtype

    # ---- optional mask
    if adj_static is not None:
        if adj_static.dim() == 2:
            mask = (adj_static > 0).to(dtype=dtype, device=device).view(1, 1, M, M)
        elif adj_static.dim() == 3:
            mask = (adj_static > 0).to(dtype=dtype, device=device).view(B, 1, M, M)
        else:
            raise ValueError("adj_static must be [M,M] or [B,M,M]")
        adj_epi = adj_prob * mask
    else:
        adj_epi = adj_prob

    diag_vals = torch.diagonal(adj_epi, dim1=-2, dim2=-1)  # [B,H,M]
    D = torch.diag_embed(diag_vals)                        # [B,H,M,M]

    col_sum = adj_epi.sum(dim=-2)                          # [B,H,M]
    W = torch.diag_embed(col_sum) - D                      # [B,H,M,M]

    A = (adj_epi.transpose(-1, -2) - D) - W                # [B,H,M,M]

    BetaDiag = torch.diag_embed(beta)                      # [B,H,M,M]
    GammaDiag = torch.diag_embed(gamma)                    # [B,H,M,M]

    tmp = (GammaDiag - A).clamp(max=clamp_max)
    I = torch.eye(M, device=device, dtype=dtype).view(1, 1, M, M)
    tmp = tmp + eps * I

    tmp_inv = torch.linalg.inv(tmp)                        # [B,H,M,M]
    ngm = BetaDiag @ tmp_inv                               # [B,H,M,M]

    x = x_last.to(device=device, dtype=dtype).view(B, 1, 1, M).expand(B, H, 1, M)
    y_epi = torch.matmul(x, ngm.transpose(-1, -2)).squeeze(-2)  # [B,H,M]
    return y_epi

class _FutureTI(nn.Module):
    def __init__(self, tid_sizes, emb_dim=4, hidden=(16,), node_specific=True, num_nodes=None):
        super().__init__()
        self.keys = list(tid_sizes.keys()) if tid_sizes else []
        self.node_specific = node_specific
        self.num_nodes = num_nodes
        self.embs = nn.ModuleDict({k: nn.Linear(K, emb_dim, bias=False) for k, K in tid_sizes.items()})
        self.total_dim = emb_dim * len(self.keys)

        def mlp(din, dout):
            layers, d = [], din
            for h in hidden: layers += [nn.Linear(d, h), nn.ReLU()]; d = h
            layers.append(nn.Linear(d, dout)); return nn.Sequential(*layers)

        self.head = mlp(self.total_dim, num_nodes if node_specific else 1)
        if node_specific: assert num_nodes is not None, "num_nodes is required when node_specific=True"

    def forward(self, states_future):
        if states_future is None or states_future.numel() == 0 or not self.keys:
            return None
        B, H, C = states_future.shape
        outs = []
        for ch, k in enumerate(self.keys):
            K = self.embs[k].in_features
            idx = states_future[..., ch].long()
            oh = F.one_hot(idx.clamp_min(0).clamp_max(K-1), K).float()
            outs.append(self.embs[k](oh))          # [B,H,emb]
        z = torch.cat(outs, dim=-1)                # [B,H,D]
        return self.head(z)                        # [B,H,N] or [B,H,1]

class _EpiHybridHead(nn.Module):
    """
    Outputs:
      - sir_incidence: [B,H,M] new cases per step (dt * beta * S * (Adj@I) / N)
      - sir_percent:   [B,H,M] percent of pop per step (sir_incidence / N * percent_scale)
      - ngm:           [B,H,M] NGM forecast

    Assumption: target series is NEW CASES (incidence) per step.
    """
    def __init__(
        self,
        in_features,
        horizon,
        population=None,        # optional
        regions=None,           # optional ("CA" or list[str] len M)
        hidden = 32,
        mlp_hidden = 8,
        target_idx = 0,
        dt = 1.0,
        percent_scale = 100.0,  # 100.0 => percent; 1.0 => fraction
        clamp_max = 1.0,
        eps = 1e-6,
        enforce_mass = True,
        learn_r0_frac = False,
        r0_init = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.horizon = horizon
        self.hidden = hidden
        self.mlp_hidden = mlp_hidden
        self.target_idx = int(target_idx)
        self.dt = float(dt)
        self.percent_scale = float(percent_scale)
        self.clamp_max = float(clamp_max)
        self.eps = float(eps)
        self.enforce_mass = bool(enforce_mass)

        self.population_spec = population
        self.regions_spec = regions

        self.gru_beta = nn.GRU(in_features, hidden, batch_first=True)
        self.gru_gamma = nn.GRU(in_features, hidden, batch_first=True)

        self.pred_beta = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, horizon),
            nn.Sigmoid(),
        )
        self.pred_gamma = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, horizon),
            nn.Sigmoid(),
        )

        if learn_r0_frac:
            # scalar -> sigmoid in (0,1)
            self.log_r0_frac = nn.Parameter(torch.tensor(float(r0_init)))
        else:
            self.log_r0_frac = None

    def _mask_adj(self, adj_prob, adj_static):
        if adj_static is None:
            return adj_prob
        B, H, M, _ = adj_prob.shape
        device, dtype = adj_prob.device, adj_prob.dtype
        if adj_static.dim() == 2:
            mask = (adj_static > 0).to(dtype=dtype, device=device).view(1, 1, M, M)
        elif adj_static.dim() == 3:
            mask = (adj_static > 0).to(dtype=dtype, device=device).view(B, 1, M, M)
        else:
            raise ValueError("adj_static must be [M,M] or [B,M,M]")
        return adj_prob * mask

    def _predict_beta_gamma(self, x):
        B, W, M, Fdim = x.shape
        x_flat = x.transpose(2, 1).contiguous().flatten(0, 1)  # [B*M, W, F]

        out_b, _ = self.gru_beta(x_flat)
        out_g, _ = self.gru_gamma(x_flat)

        last_b = out_b[:, -1, :]
        last_g = out_g[:, -1, :]

        beta = self.pred_beta(last_b).view(B, M, self.horizon).transpose(1, 2)   # [B,H,M]
        gamma = self.pred_gamma(last_g).view(B, M, self.horizon).transpose(1, 2) # [B,H,M]
        return beta, gamma

    def _expand_N(self, B, M, device, dtype):
        N_m = resolve_population_2019(
            M=M,
            population=self.population_spec,
            regions=self.regions_spec,
            device=device,
            dtype=dtype,
        )  # [M]
        return N_m.view(1, M).expand(B, M)  # [B,M]

    def _sir_incidence_rollout(self, x, adj_prob, beta, gamma):
        """
        Network-mixed latent SIR rollout producing NEW CASES per step.

        Definitions:
        I_eff(h) = Adj[h] @ I(h)   (network infectious pressure)
        new_inf_rate(h) = beta[h] * S(h) * I_eff(h) / N
        new_cases(h) = dt * new_inf_rate(h)

        Args:
        x:        [B,W,M,F]   input window, where x[:, -1, :, target_idx] is last observed NEW CASES
        adj_prob: [B,H,M,M]   (soft) adjacency weights per horizon step
        beta:     [B,H,M]     per-node transmission factor in (0,1)
        gamma:    [B,H,M]     per-node recovery factor in (0,1)

        Returns:
        cases: [B,H,M] new cases per step
        N_bm:  [B,M]   population per node
        """
        B, W, M, _ = x.shape
        device, dtype = x.device, x.dtype

        # population per node (broadcasted to [B,M])
        N_bm = self._expand_N(B, M, device, dtype)  # [B,M]

        # last observed incidence (new cases per step)
        C0 = x[:, -1, :, self.target_idx].to(device=device, dtype=dtype)
        C0 = torch.clamp(C0, min=0.0)

        # --- initialize latent compartments from incidence
        # C0 ≈ dt * beta0 * S0 * I0 / N, assume S0≈N -> I0 ≈ C0 / (dt*beta0)
        beta0 = torch.clamp(beta[:, 0, :], min=1e-4)  # [B,M], avoid divide-by-zero
        I0 = C0 / (self.dt * beta0 + 1e-12)          # [B,M]
        I0 = I0.clamp(min=0.0)
        I0 = torch.minimum(I0, N_bm)                 # tensor-safe upper bound

        # optional R0 as fraction of remaining mass
        if self.log_r0_frac is None:
            R0 = torch.zeros_like(I0)
        else:
            r0 = torch.sigmoid(self.log_r0_frac)     # scalar in (0,1)
            R0 = r0 * torch.clamp(N_bm - I0, min=0.0)

        S0 = torch.clamp(N_bm - I0 - R0, min=0.0)

        # current state
        S, I, R = S0, I0, R0

        cases = []
        for h in range(self.horizon):
            # --- network mixing: I_eff = Adj @ I
            adj_h = adj_prob[:, h, :, :]  # [B,M,M]
            I_eff = torch.bmm(adj_h, I.unsqueeze(-1)).squeeze(-1)  # [B,M]
            I_eff = I_eff.clamp(min=0.0)

            # --- infection flow
            beta_h = beta[:, h, :].clamp(min=0.0)
            gamma_h = gamma[:, h, :].clamp(min=0.0)

            new_inf_rate = beta_h * S * I_eff / (N_bm + 1e-12)     # per unit time
            new_inf_rate = new_inf_rate.clamp(min=0.0)

            new_cases = self.dt * new_inf_rate                     # per step (e.g., per day)
            cases.append(new_cases)

            # --- Euler update
            dS = -new_inf_rate
            dI = new_inf_rate - gamma_h * I
            dR = gamma_h * I

            S = (S + self.dt * dS).clamp(min=0.0)
            I = (I + self.dt * dI).clamp(min=0.0)
            R = (R + self.dt * dR).clamp(min=0.0)

            # enforce conservation S+I+R = N (helps prevent drift)
            if self.enforce_mass:
                total = S + I + R
                scale = N_bm / (total + 1e-12)
                S = S * scale
                I = I * scale
                R = R * scale

        cases = torch.stack(cases, dim=1)  # [B,H,M]
        return cases, N_bm

    def forward(self, x, adj_prob, adj_static=None, mode="sir_incidence"):
        """
        mode:
          - "sir_incidence": [B,H,M] new cases per step
          - "sir_percent":   [B,H,M] percent of population per step
          - "ngm":           [B,H,M]
        """
        assert x.dim() == 4, "x must be [B,W,M,F]"
        assert adj_prob.dim() == 4, "adj_prob must be [B,H,M,M]"
        assert adj_prob.shape[1] == self.horizon, "adj_prob horizon mismatch"

        adj_prob_masked = self._mask_adj(adj_prob, adj_static)
        beta, gamma = self._predict_beta_gamma(x)

        if mode in ("sir_incidence", "sir_percent"):
            sir_inc, N_bm = self._sir_incidence_rollout(x, adj_prob_masked, beta, gamma)  # [B,H,M], [B,M]
            sir_pct = (sir_inc / (N_bm.unsqueeze(1) + 1e-12)) * self.percent_scale        # [B,H,M]

        if mode == "sir_incidence":
            return sir_inc
        if mode == "sir_percent":
            return sir_pct
        if mode == "ngm":
            x_last = x[:, -1, :, self.target_idx]  # last observed series (assumed incidence)
            return compute_epi_ngm_forecast(
                adj_prob=adj_prob_masked,
                beta=beta,
                gamma=gamma,
                x_last=x_last,
                adj_static=adj_static,
                clamp_max=self.clamp_max,
                eps=self.eps,
            )

        raise ValueError(f"Unknown mode: {mode}")

class _EpiRegLoss(nn.Module):
    def __init__(self, scale=0.5, loss='mse'):
        super().__init__()
        self.scale = float(scale)
        self.loss = str(loss)

    def forward(self, epi_out, target):
        if self.loss == 'mse':
            reg = F.mse_loss(epi_out, target)
        elif self.loss == 'l1':
            reg = F.l1_loss(epi_out, target)
        elif self.loss == 'smooth_l1':
            reg = F.smooth_l1_loss(epi_out, target)
        else:
            raise ValueError(f"Unknown epi_reg_loss: {self.loss}")
        return self.scale * reg

class BaseModel(nn.Module):
    def __init__(self, device = 'cpu', 
                 use_future_ti=False, tid_sizes=None, emb_dim=4, ti_hidden=(16,), node_specific=True, num_nodes=None):
        super(BaseModel, self).__init__()
        self.device = device
        self.future_ti = _FutureTI(tid_sizes, emb_dim, ti_hidden, node_specific, num_nodes).to(device) \
                         if (use_future_ti and tid_sizes) else None

    def predict_samples(self, feature, graph=None, states=None, dynamic_graph=None, n_samples=100, filtered = False):
        """
        Default: additive Gaussian noise with std estimated from training residuals.
        Returns: (S, ...) where ... is predict() shape.
        """
        base = self.predict(feature, graph=graph, states=states, dynamic_graph=dynamic_graph).to(self.device)
        if filtered == True:
            noise_std = self._noise_std_filtered
        else:
            noise_std = self._noise_std
        if noise_std is None:
            # no estimate available -> tiny noise instead of identical copies
            eps_scale = 1e-3 * (base.std() + 1.0)
            noise = torch.randn((n_samples,) + tuple(base.shape), device=self.device) * eps_scale
            return base.unsqueeze(0) + noise
        noise_std = noise_std.to(self.device)
        # broadcast noise_std to base.shape
        while noise_std.dim() < base.dim():
            noise_std = noise_std.unsqueeze(0)  # match leading batch dim
        noise = torch.randn((n_samples,) + tuple(base.shape), device=self.device) * noise_std.unsqueeze(0)
        return base.unsqueeze(0) + noise

    def predict_quantiles(
        self,
        feature,
        quantiles,
        graph=None,
        states=None,
        dynamic_graph=None,
        n_samples=100,
        filtered = False
    ):
        samples = self.predict_samples(
            feature, graph=graph, states=states, dynamic_graph=dynamic_graph, n_samples=n_samples, filtered = filtered
        ).to(self.device)
        q_tensor = torch.tensor(quantiles, device=self.device, dtype=samples.dtype)
        q = torch.quantile(samples, q=q_tensor, dim=0)
        return q.cpu()

    def fit(self, 
            train_input, 
            train_target, 
            train_states=None, 
            train_graph=None, 
            train_dynamic_graph=None,
            val_input=None, 
            val_target=None,
            val_states=None, 
            val_graph= None, 
            val_dynamic_graph=None,
            loss='mse', 
            use_epi_reg=False,
            epi_reg_loss='mse',
            epi_hidden=8,
            epi_mode="sir_incidence",   # "sir_incidence" | "sir_percent" | "ngm"
            epi_percent_scale=100.0,    # percent output scaling for sir_percent
            epi_population=None,        # optional scalar / [M] list/tensor
            epi_regions=None,           # optional "CA" or list[str] length M
            epochs=1000, 
            batch_size=10,
            lr=1e-3, 
            initialize=True, 
            verbose=False, 
            patience=100, 
            **kwargs):
        if initialize:
            self.initialize()
        self._setup_epi_reg_from_data(
            train_input, train_target,
            use_epi_reg=use_epi_reg,
            epi_reg_loss=epi_reg_loss,
            epi_hidden=epi_hidden,
            epi_mode=epi_mode,
            epi_percent_scale=epi_percent_scale,
            epi_regions=epi_regions,
            epi_population=epi_population,
            target_idx=0,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        loss_fn = get_loss(loss)

        training_losses = []
        validation_losses = []
        early_stopping = patience
        best_val = float('inf')
        best_weights = deepcopy(self.state_dict())
        for epoch in tqdm(range(epochs)):
            # train one epoch
            # import ipdb; ipdb.set_trace()
            loss = self.train_epoch(optimizer=optimizer, 
                                    loss_fn=loss_fn, 
                                    feature=train_input, 
                                    states=train_states, 
                                    graph=train_graph, 
                                    dynamic_graph=train_dynamic_graph, 
                                    target=train_target, 
                                    batch_size=batch_size, 
                                    device=self.device)
            training_losses.append(loss)
            # validate
            if val_input is not None and val_input.numel():
                val_loss, output = self.evaluate(loss_fn=loss_fn, 
                                                feature=val_input, 
                                                graph=val_graph, 
                                                dynamic_graph=val_dynamic_graph,
                                                target=val_target, 
                                                states=val_states, 
                                                device=self.device)
                validation_losses.append(val_loss)
                if val_loss is not None and best_val > val_loss:
                    best_val = val_loss
                    self.best_output = output
                    best_weights = deepcopy(self.state_dict())
                    patience = early_stopping
                else:
                    patience -= 1

                if epoch > early_stopping and patience <= 0:
                    print("Early stopping at epoch: ", epoch)
                    break

                if epoch%10 == 0:
                    print(f"######### epoch:{epoch}")
                    print("Training loss: {}".format(training_losses[-1]))
                    print("Validation loss: {}".format(validation_losses[-1]))
            else:
                validation_losses.append(None)
                best_weights = deepcopy(self.state_dict())
                print(f"######### epoch:{epoch}")
                print("Training loss: {}".format(training_losses[-1]))
                print("Validation loss: {}".format(validation_losses[-1]))
            

        print("\n")
        print("Final Training loss: {}".format(training_losses[-1]))
        print("Final Validation loss: {}".format(validation_losses[-1]))

        plt.figure()
        plt.plot(training_losses, label="train")
        plt.plot(validation_losses, label="val")
        plt.legend()
        plt.savefig("st_loss.png")
        plt.show()
        
        self.load_state_dict(best_weights)
        self._estimate_noise_std(train_input, train_target, train_states, train_graph, train_dynamic_graph)

    def _estimate_noise_std(self, feature, target, states=None, graph=None, dynamic_graph=None, iqr_mult=1.5, exclude_zeros=True):
        self.eval()
        with torch.no_grad():
            pred = self.predict(feature=feature,
                                graph=graph,
                                states=states,
                                dynamic_graph=dynamic_graph).to(self.device)
            target = target.to(self.device)
            pred = pred.reshape_as(target)
            diff = pred - target
            # if not filtered:
            self._noise_std = diff.std(dim=0, unbiased=True)
            if torch.all(self._noise_std == 0):
                self._noise_std = diff.std() * torch.ones_like(self._noise_std)
            # if filtered:
            mask_elem = torch.isfinite(target)
            if exclude_zeros:
                mask_elem &= (target != 0)
            if mask_elem.sum() == 0:
                resid = diff
            else:
                v = target[mask_elem]
                q1, q3 = torch.quantile(v, torch.tensor([0.25, 0.75], device=target.device))
                iqr = q3 - q1
                if iqr == 0:
                    lower, upper = q1, q3
                else:
                    lower = q1 - iqr_mult * iqr
                    upper = q3 + iqr_mult * iqr
                mask_elem &= (target >= lower) & (target <= upper)
                mask_sample = mask_elem.view(mask_elem.shape[0], -1).all(dim=1)
                if mask_sample.sum() == 0:
                    resid = diff
                else:
                    resid = diff[mask_sample]
            self._noise_std_filtered = resid.std(dim=0, unbiased=True)
            if torch.all(self._noise_std_filtered == 0):
                self._noise_std_filtered = resid.std() * torch.ones_like(self._noise_std_filtered)

    def _apply_future_ti(self, y, states):
        # states: [B,H,C], y: [B,N,H] or [B,H,N]
        if self.future_ti is None or states is None or states.numel() == 0:
            return y
        delta = self.future_ti(states)  # [B,H,N] or [B,H,1]
        if delta is None:
            return y
        # Align shapes to [B,H,N] for addition
        B, Hs, _ = states.shape
        if y.dim() != 3:
            return y
        y_is_BNH = (y.shape[2] == Hs)      # True if [B,N,H]
        y_hn = y.transpose(1, 2) if y_is_BNH else y  # -> [B,H,N]
        if delta.size(-1) == 1:
            delta = delta.expand(-1, -1, y_hn.size(-1))
        y_hn = y_hn + delta
        return y_hn.transpose(1, 2) if y_is_BNH else y_hn

    def _setup_epi_reg_from_data(
        self,
        train_input,
        train_target,
        use_epi_reg=False,
        epi_reg_loss="mse",
        epi_hidden=8,
        epi_mode="sir_incidence",   # "sir_incidence" | "sir_percent" | "ngm"
        epi_percent_scale=100.0,    # percent output scaling for sir_percent
        epi_population=None,        # optional scalar / [M] list/tensor
        epi_regions=None,           # optional "CA" or list[str] length M
        epi_dt=1.0,                 # step size
        target_idx=0,               # which feature is the observed cases series
    ):
        if not use_epi_reg:
            self.epi_head = None
            self.epi_reg = None
            self.epi_mode = None
            return

        _, _, M, Fdim = train_input.shape

        if train_target.dim() != 3:
            raise ValueError("train_target must be 3D")
        if train_target.shape[1] == M and train_target.shape[2] != M:
            H = train_target.shape[2]  # [B,M,H]
        else:
            H = train_target.shape[1]  # [B,H,M]

        self.epi_mode = epi_mode

        self.epi_head = _EpiHybridHead(
            in_features=Fdim,
            horizon=H,
            population=epi_population,
            regions=epi_regions,
            hidden=epi_hidden,
            mlp_hidden=epi_hidden,
            target_idx=target_idx,
            dt=epi_dt,
            percent_scale=epi_percent_scale,
            clamp_max=1.0,
            eps=1e-6,
            enforce_mass=True,
        ).to(self.device)

        scale = float(use_epi_reg) if isinstance(use_epi_reg, (int, float)) else 1.0
        self.epi_reg = _EpiRegLoss(scale=scale, loss=epi_reg_loss).to(self.device)

    def _apply_epi_reg_loss(self, base_loss, feature, graph, dynamic_graph, target):
        if self.epi_head is None or self.epi_reg is None:
            return base_loss

        B, W, M, Fdim = feature.shape
        device = feature.device

        # static adjacency for masking (optional)
        adj_static = graph.to(device) if graph is not None else None

        # build adj_prob [B,H,M,M]
        H = self.epi_head.horizon

        if dynamic_graph is not None:
            dg = dynamic_graph.to(device)
            if dg.dim() == 4:
                # assume [B,H,M,M] (or [B,1,M,M] -> expand)
                adj_prob = F.softmax(dg, dim=-1)
                if adj_prob.shape[1] == 1 and H > 1:
                    adj_prob = adj_prob.expand(B, H, M, M)
                elif adj_prob.shape[1] != H:
                    # best effort: truncate or pad by repeating last
                    if adj_prob.shape[1] > H:
                        adj_prob = adj_prob[:, :H]
                    else:
                        last = adj_prob[:, -1:].expand(B, H - adj_prob.shape[1], M, M)
                        adj_prob = torch.cat([adj_prob, last], dim=1)
            elif dg.dim() == 3:
                # [B,M,M] -> expand to [B,H,M,M]
                adj_prob = F.softmax(dg, dim=-1).view(B, 1, M, M).expand(B, H, M, M)
            else:
                adj_prob = None
        else:
            adj_prob = None

        if adj_prob is None:
            # fallback to static graph
            if adj_static is None:
                return base_loss  # cannot compute epi reg without any graph
            adj_prob = F.softmax(adj_static, dim=-1).view(1, 1, M, M).expand(B, H, M, M)

        # compute epi output in chosen mode
        epi_out = self.epi_head(feature, adj_prob, adj_static=adj_static, mode=self.epi_mode)

        # shape align: allow target [B,M,H]
        if epi_out.shape != target.shape and epi_out.dim() == 3 and epi_out.transpose(1, 2).shape == target.shape:
            epi_out = epi_out.transpose(1, 2)

        return base_loss + self.epi_reg(epi_out, target)

    def train_epoch(self, optimizer, loss_fn, feature, states=None, graph=None, dynamic_graph=None, target=None, batch_size=1, device='cpu'):
        """
        Trains one epoch with the given data.
        :param feature: Training features of shape (num_samples, num_nodes,
        num_timesteps_train, num_features).
        :param target: Training targets of shape (num_samples, num_nodes,
        num_timesteps_predict).
        :param batch_size: Batch size to use during training.
        :return: Average loss for this epoch.
        """
        permutation = torch.randperm(feature.shape[0])

        epoch_training_losses = []
        for i in range(0, feature.shape[0], batch_size):
            self.train()
            optimizer.zero_grad()
            
            indices = permutation[i:i + batch_size]
            X_batch, y_batch = feature[indices], target[indices]

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if states is not None:
                X_states = states[indices]
                X_states = X_states.to(device)
            else:
                X_states = None
            
            if dynamic_graph is not None:
                batch_graph = dynamic_graph[indices]
                batch_graph = batch_graph.to(device)
            else:
                batch_graph = None
            
            if graph is not None:
                graph = graph.to(device)
            out = self.forward(X_batch, graph, X_states, batch_graph)
            out = self._apply_future_ti(out, X_states)
            loss = loss_fn(out, y_batch)
            # import ipdb; ipdb.set_trace()
            loss = self._apply_epi_reg_loss(loss, X_batch, graph, batch_graph, y_batch)
            loss.backward()
            optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy())
        return sum(epoch_training_losses)/len(epoch_training_losses)
    
    def evaluate(self, loss_fn, feature, graph = None, dynamic_graph=None, target = None, states = None, device = 'cpu'):
        with torch.no_grad():
            self.eval()
            feature = feature.to(device=device)
            target = target.to(device=device)

            if graph is not None:
                graph = graph.to(device)

            if dynamic_graph is not None:
                dynamic_graph = dynamic_graph.to(device)
            
            if states is not None:
                states = states.to(device)

            out = self.forward(feature, graph, states, dynamic_graph)
            out = self._apply_future_ti(out, states)
            val_loss = loss_fn(out, target)
            val_loss = self._apply_epi_reg_loss(val_loss, feature, graph, dynamic_graph, target)
            val_loss = val_loss.detach().cpu().item()
            
            return val_loss, out

    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """
        Returns
        -------
        torch.FloatTensor
        """
        with torch.no_grad():
            self.eval()
            if graph is not None:
                graph = graph.to(self.device)

            if dynamic_graph is not None:
                dynamic_graph = dynamic_graph.to(self.device)
            
            if states is not None:
                states = states.to(self.device)
            
            if feature is not None:
                feature = feature.to(self.device)
            # import ipdb; ipdb.set_trace()
            result = self.forward(feature, graph, states, dynamic_graph)
            result = self._apply_future_ti(result, states)
        return result.detach().cpu()

class BaseTemporalModel(nn.Module):
    def __init__(self, device = 'cpu'):
        super(BaseTemporalModel, self).__init__()
        self.device = device

    def fit(self, 
            train_input, 
            train_target, 
            train_states=None, 
            train_graph=None, 
            train_dynamic_graph=None,
            val_input=None, 
            val_target=None,
            val_states=None, 
            val_graph= None, 
            val_dynamic_graph=None,
            loss='mse', 
            epochs=1000, 
            batch_size=10,
            lr=1e-3, 
            initialize=True, 
            verbose=False, 
            patience=100, 
            **kwargs):
        if initialize:
            self.initialize()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = get_loss(loss)

        training_losses = []
        validation_losses = []
        early_stopping = patience
        best_val = float('inf')
        for epoch in tqdm(range(epochs)):
            
            loss = self.train_epoch(optimizer = optimizer, loss_fn = loss_fn, feature = train_input,  target = train_target, batch_size = batch_size, device = self.device)
            training_losses.append(loss)
            if val_input is not None and val_input.numel():
                val_loss, output = self.evaluate(loss_fn = loss_fn, feature = val_input,  target = val_target, device = self.device)
                validation_losses.append(val_loss)

                if best_val > val_loss:
                    best_val = val_loss
                    self.output = output
                    best_weights = deepcopy(self.state_dict())
                    patience = early_stopping
                else:
                    patience -= 1

                if epoch > early_stopping and patience <= 0:
                    break

                if verbose and epoch%50 == 0:
                    print(f"######### epoch:{epoch}")
                    print("Training loss: {}".format(training_losses[-1]))
                    print("Validation loss: {}".format(validation_losses[-1]))
            else:
                validation_losses.append(None)
                best_weights = deepcopy(self.state_dict())
                if verbose and epoch%50 == 0:
                    print(f"######### epoch:{epoch}")
                    print("Training loss: {}".format(training_losses[-1]))
                    print("Validation loss: {}".format(validation_losses[-1]))

        print("\n")
        print("Final Training loss: {}".format(training_losses[-1]))
        print("Final Validation loss: {}".format(validation_losses[-1]))

        self.load_state_dict(best_weights)

        
    def train_epoch(self, optimizer, loss_fn, feature, target = None, batch_size = 1, device = 'cpu'):
        """
        Trains one epoch with the given data.
        :param feature: Training features of shape (num_samples, num_nodes,
        num_timesteps_train, num_features).
        :param target: Training targets of shape (num_samples, num_nodes,
        num_timesteps_predict).
        :param batch_size: Batch size to use during training.
        :return: Average loss for this epoch.
        """
        permutation = torch.randperm(feature.shape[0])

        epoch_training_losses = []
        for i in range(0, feature.shape[0], batch_size):
            self.train()
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            X_batch, y_batch = feature[indices], target[indices]
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)
            
            out = self.forward(X_batch)
            loss = loss_fn(out.reshape(y_batch.shape), y_batch)
            loss.backward()
            optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy())
        return sum(epoch_training_losses)/len(epoch_training_losses)
    
    def evaluate(self, loss_fn, feature, target = None, device = 'cpu'):
        with torch.no_grad():
            self.eval()
            feature = feature.to(device=device)
            target = target.to(device=device)

            out = self.forward(feature)
            val_loss = loss_fn(out.reshape(target.shape), target)
            val_loss = val_loss.detach().cpu().numpy().item()
            
            return val_loss, out

    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """
        Returns
        -------
        torch.FloatTensor
        """
        self.eval()
        result = self.forward(feature.to(self.device))

        return result.detach().cpu()