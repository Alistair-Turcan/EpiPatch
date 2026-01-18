# EpiPatch

A compact benchmarking + patching repo for **spatiotemporal epidemic forecasting** (multi-region, multi-horizon).

It includes:
- **Baselines** (naïve "today as future", repeat-last, ARIMA)
- **ML models** (DLinear, AGCRN, ColaGNN, DCRNN, EpiGNN, GraphWaveNet, MTGNN, STGCN, GTS, StemGNN, STNorm, EARTH)
- Optional "patches" in `BaseModel`:
  - **Future Time Indicators (TID)** (future-known calendar covariates like day-of-week)
  - **Filtered** errors and sampling
  - **Epidemiological (ODE) regularization** (latent SIR incidence + NGM propagation)

---

## Data preparation

This snippet is the *canonical* preprocessing used by the experiments below.

```python
lookback = 28
horizon = 7
epochs = 100
train_rate = 0.6
val_rate = 0.2
region_idx = None
permute = False

dow_flag = True
woy_flag = False

data_df = pd.read_csv("rawData/processed/JHUcase.csv", index_col = 0)
data_df.index = pd.to_datetime(data_df.index)
adj_df = pd.read_csv("rawData/processed/JHUcase_adj.csv", index_col = 0)

dataset = UniversalDataset()
data = np.expand_dims(data_df.values, axis=-1)
dataset.x = torch.FloatTensor(data)
dataset.y = torch.FloatTensor(data)[:, :, 0]
dataset.graph = torch.FloatTensor(adj_df.to_numpy())
train_dataset, val_dataset, test_dataset = dataset.ganerate_splits(train_rate=train_rate, val_rate=val_rate)
adj = train_dataset['graph']
adj = adj.type(torch.float)
num_nodes = adj.shape[0]

if dow_flag:
    dow = torch.as_tensor(data_df.index.dayofweek.values, dtype=torch.long)
    if woy_flag:
        woy = torch.as_tensor(data_df.index.isocalendar().week.values - 1, dtype=torch.long)
        dataset.states = torch.stack([dow, woy], dim=-1)
        tid_s={'dow': 7, 'woy': 53}
    else:
        dataset.states = torch.stack([dow], dim=-1)
        tid_s={'dow': 7}
elif woy_flag:
    woy = torch.as_tensor(data_df.index.isocalendar().week.values - 1, dtype=torch.long)
    dataset.states = torch.stack([woy], dim=-1)
    tid_s={'woy': 53}

train_dataset, val_dataset, test_dataset = dataset.ganerate_splits(train_rate=train_rate, val_rate=val_rate)
adj = train_dataset["graph"].type(torch.float)

train_input, train_target, train_states_past, train_states_future, train_adj = generate_dataset(
    X=train_dataset["features"],
    Y=train_dataset["target"],
    states=train_dataset["states"],
    dynamic_adj=train_dataset["dynamic_graph"],
    lookback_window_size=lookback,
    horizon_size=horizon,
    ahead=0,
    permute=permute,
)

val_input, val_target, val_states_past, val_states_future, val_adj = generate_dataset(
    X=val_dataset["features"],
    Y=val_dataset["target"],
    states=val_dataset["states"],
    dynamic_adj=val_dataset["dynamic_graph"],
    lookback_window_size=lookback,
    horizon_size=horizon,
    ahead=0,
    permute=permute,
)

test_input, test_target, test_states_past, test_states_future, test_adj = generate_dataset(
    X=test_dataset["features"],
    Y=test_dataset["target"],
    states=test_dataset["states"],
    dynamic_adj=test_dataset["dynamic_graph"],
    lookback_window_size=lookback,
    horizon_size=horizon,
    ahead=0,
    permute=permute,
)

train_split = {
    "features": train_input,
    "targets": train_target,
    "states": train_states_future,  # future-only time IDs
    "dynamic_graph": train_adj,
}
val_split = {
    "features": val_input,
    "targets": val_target,
    "states": val_states_future,
    "dynamic_graph": val_adj,
}
test_split = {
    "features": test_input,
    "targets": test_target,
    "states": test_states_future,
    "dynamic_graph": test_adj,
}
```

- `lookback`: how many past time points are fed into the model per sample (e.g., 28).
- `horizon`: how many future time points to predict per sample (e.g., 7).
- `train_rate`, `val_rate`: temporal split ratios (test is the remainder).
- `tid_s` stores the categorical vocabulary sizes (e.g., `dow:7`, `woy:53`) for the TID patch.

---

## Baselines

### Baseline 1: "today as prediction of k-days later"

```python
extended_input, extended_target, extended_states_past, extended_states_future, extended_adj = generate_dataset(
    X=dataset.x,
    Y=dataset.y,
    lookback_window_size=lookback,
    horizon_size=horizon,
    ahead=0,
    permute=permute,
)

for fw_days in range(1, horizon + 1):
    targets = test_split["targets"]
    mse = metrics.get_MSE(extended_target[-(fw_days + targets.shape[0]) : -fw_days], targets)
    mae = metrics.get_MAE(extended_target[-(fw_days + targets.shape[0]) : -fw_days], targets)
    rmse = metrics.get_RMSE(extended_target[-(fw_days + targets.shape[0]) : -fw_days], targets)
    mse_filtered = metrics.get_MSE_filtered(extended_target[-(fw_days + targets.shape[0]) : -fw_days], targets)
    mae_filtered = metrics.get_MAE_filtered(extended_target[-(fw_days + targets.shape[0]) : -fw_days], targets)
    medse = metrics.get_medSE(extended_target[-(fw_days + targets.shape[0]) : -fw_days], targets)
    medae = metrics.get_medAE(extended_target[-(fw_days + targets.shape[0]) : -fw_days], targets)
```

### Baseline 2: "repeat last value"

```python
targets = test_split["targets"]
pred = test_split["features"][:, -1, :, 0].unsqueeze(1).repeat(1, horizon, 1)

mse = metrics.get_MSE(pred, targets)
mae = metrics.get_MAE(pred, targets)
rmse = metrics.get_RMSE(pred, targets)
mse_filtered = metrics.get_MSE_filtered(pred, targets)
mae_filtered = metrics.get_MAE_filtered(pred, targets)
medse = metrics.get_medSE(pred, targets)
medae = metrics.get_medAE(pred, targets)
```

### Baseline 3: ARIMA

```python
from EpiLearnSpatialTemporal.ARIMA import ARIMA

model = ARIMA(num_timesteps_input=lookback, num_timesteps_output=horizon, num_features=1, device="cpu")
```

---

## ML models (direct instantiation)

All models below share the same `fit(...)` calling style shown in the examples. Many constructors include additional hyperparameters; the snippets show the most common ones used in this repo.

### DLinear

```python
from EpiLearnSpatialTemporal.Dlinear import DlinearModel

model = DlinearModel(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    num_features=1,
    num_nodes=num_nodes,

    # TID patch (optional)
    use_future_ti=True,
    tid_sizes=tid_s,       # e.g., {'dow': 7} or {'dow': 7, 'woy': 53}
    emb_dim=4,
    ti_hidden=(8,),
    node_specific=True,

    device='cpu'
)
```

### AGCRN

```python
from EpiLearnSpatialTemporal.AGCRN import AGCRN

model = AGCRN(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    adj_m=adj,
    num_nodes=num_nodes,
    num_features=1,
    device='cpu',
    rnn_units=16,
    nlayers=2,
    embed_dim=8,
    cheb_k=2,
    # ...
)
```

### ColaGNN

```python
from EpiLearnSpatialTemporal.ColaGNN import ColaGNN

model = ColaGNN(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    adj_m=adj,
    num_nodes=num_nodes,
    num_features=1,
    device='cpu',
    nhid=16,
    n_layer=2,
    # ...
)
```

### DCRNN

```python
from EpiLearnSpatialTemporal.DCRNN import DCRNN

model = DCRNN(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    adj_m=adj,
    num_nodes=num_nodes,
    num_features=1,
    device='cpu',
    max_diffusion_step=3,
    # ...
)
```

### EpiGNN

```python
from EpiLearnSpatialTemporal.EpiGNN import EpiGNN

model = EpiGNN(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    adj_m=adj,
    num_nodes=num_nodes,
    num_features=1,
    device='cpu',
    k=5,
    hidA=32,
    hidR=4,
    hidP=1,
    n_layer=2,
    dropout=0.2,
    # ...
)
```

### GraphWaveNet

```python
from EpiLearnSpatialTemporal.GraphWaveNet import GraphWaveNet

model = GraphWaveNet(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    adj_m=adj,
    num_nodes=num_nodes,
    num_features=1,
    device='cpu',
    residual_channels=4,
    dilation_channels=4,
    skip_channels=32,
    end_channels=64,
    kernel_size=2,
    blocks=4,
    nlayers=8,
    # ...
)
```

### MTGNN

```python
from EpiLearnSpatialTemporal.MTGNN import MTGNN

model = MTGNN(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    adj_m=adj,
    num_nodes=num_nodes,
    num_features=1,
    device='cpu',
    gcn_depth=2,
    dropout=0.2,
    subgraph_size=3,
    node_dim=8,
    dilation_exponential=1,
    conv_channels=8,
    residual_channels=4,
    skip_channels=8,
    end_channels=32,
    layers=3,
    propalpha=0.05,
    tanhalpha=3,
    # ...
)
```

### STGCN

```python
from EpiLearnSpatialTemporal.STGCN import STGCN

model = STGCN(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    adj_m=adj,
    num_nodes=num_nodes,
    num_features=1,
    device='cpu',
    nhids=16,
    # ...
)
```

### GTS

```python
from EpiLearnSpatialTemporal.GTS import GTS

model = GTS(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    adj_m=adj,
    num_nodes=num_nodes,
    num_features=1,
    device='cpu',
    rnn_units=32,
    max_diffusion_step=2,
    # ...
)
```

### StemGNN

```python
from EpiLearnSpatialTemporal.StemGNN import StemGNN

model = StemGNN(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    adj_m=adj,
    num_nodes=num_nodes,
    num_features=1,
    device='cpu',
    stack_cnt=2,
    multi_layer=4,
    dropout_rate=0.2,
    leaky_rate=0.2,
    # ...
)
```

### STNorm

```python
from EpiLearnSpatialTemporal.STNorm import STNorm

model = STNorm(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    adj_m=adj,
    num_nodes=num_nodes,
    num_features=1,
    device='cpu',
    channels=8,
    kernel_size=2,
    blocks=8,
    layers=2,
    # ...
)
```

### EARTH (DTW matrix required)

EARTH requires a DTW distance matrix (cached as a `.npy` file).

```python
from fastdtw import fastdtw
dataset_name = "JHUcase"
cache_path = f"dtw_{dataset_name}.npy"

if os.path.exists(cache_path):
    dtw_matrix = np.load(cache_path)
    print(f"Loaded DTW matrix from {cache_path}")
else:
    num_nodes = train_dataset["features"].shape[1]
    data_mean = train_dataset["features"].reshape(train_dataset["features"].shape[0], num_nodes, 1)
    dtw_matrix = np.zeros((num_nodes, num_nodes))
    for i in tqdm(range(num_nodes)):
        for j in range(i, num_nodes):
            dtw_distance, _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
            dtw_matrix[i][j] = dtw_distance
    for i in range(num_nodes):
        for j in range(i):
            dtw_matrix[i][j] = dtw_matrix[j][i]
    np.save(cache_path, dtw_matrix)
    print(f"Saved DTW matrix to {cache_path}")

from EpiLearnSpatialTemporal.EARTH import EARTH

model = EARTH(
    dtw_matrix=dtw_matrix,
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    adj_m=adj,
    num_nodes=num_nodes,
    num_features=1,
    device="cpu",
    # ...
)
```

---

## Training: `BaseModel.fit(...)` (common for most models)

Most models in this repo inherit from `BaseModel`, so training looks like:

```python
model.fit(
    train_input=train_split["features"],
    train_target=train_split["targets"],
    train_states=train_split["states"],
    train_graph=adj,
    train_dynamic_graph=train_split["dynamic_graph"],
    val_input=val_split["features"],
    val_target=val_split["targets"],
    val_states=val_split["states"],
    val_graph=adj,
    val_dynamic_graph=val_split["dynamic_graph"],
    loss="mse",
    epochs=epochs,
    batch_size=10,
    lr=1e-3,
)
```

---

## Patch options (built into `BaseModel`)

### 1) Future Time Indicators (TID): `use_future_ti=True`

**Idea.** If future covariates like weekday/week-of-year are known at inference time, you can add a learned seasonal offset to the model’s horizon forecast. Set `use_future_ti=True` and pass `tid_sizes` (e.g., `{'dow': 7}`) to turn it on.

---

### 2) ODE / epidemiological regularization

Enable by setting `use_epi_reg`. The training loss becomes base loss plus adds reg term,
where `epi_reg_loss` chooses `mse`($\frac{1}{n}\sum (\hat{y}-y)^2$), `l1`($\frac{1}{n}\sum |\hat{y}-y|$), or `smooth_l1`(Huber-like robust loss (quadratic near 0, linear in tails)).

#### Modes: `epi_mode`

- **`sir_incidence`**(EINNs-inspired): runs a simple **SIR-style rollout** and outputs **new cases per step**. Let `P` be the (soft) graph mixing matrix, so infections can "flow" across regions:
  $$
  I_{\text{mix}} = P\,I
  $$
  New cases (incidence) at each step are computed by the infection flow:
  $$
  \text{new\_cases} = \Delta t \cdot \beta \cdot \frac{S}{N} \cdot I_{\text{mix}}
  $$
  with Euler updates:
  $$
  S \leftarrow S - \text{new\_cases},\quad I \leftarrow I + \text{new\_cases} - \Delta t\cdot \gamma \cdot I,\quad R \leftarrow R + \Delta t\cdot \gamma \cdot I
  $$

- **`sir_percent`**(EINNs-inspired): same as `sir_incidence`, but returns **per-capita new cases**:
  $$
  \text{percent\_cases} = s \cdot \frac{\text{new\_cases}}{N}
  $$
  where `epi_percent_scale = s` (e.g., `100` for %, `1e5` for per-100k).

- **`ngm`**(Epi-Cola-GNN-inspired): builds a **Next-Generation-Matrix-like operator** and propagates the last observed cases forward. Think of `K` as a "who-infects-who across regions" matrix:
  $$
  K = \mathrm{diag}(\beta)\,\big(\mathrm{diag}(\gamma) - A\big)^{-1}
  $$
  Then the epi forecast is:
  $$
  \text{epi\_pred} = \text{last\_cases} \cdot K^{\top}
  $$

**Population input.**
- `epi_population`: scalar or per-node vector, or
- `epi_regions`: list of region codes (e.g., US states) to auto-resolve from the built-in population table.

---

## Example: DLinear + SIR-incidence epi regularization

```python
from EpiLearnSpatialTemporal.Dlinear import DlinearModel

fix_seed(42)

model = DlinearModel(
    num_timesteps_input=lookback,
    num_timesteps_output=horizon,
    num_features=1,
    num_nodes=num_nodes,
    use_future_ti=False,
    tid_sizes=None,
    emb_dim=4,
    ti_hidden=(8,),
    device='cpu'
)

model.fit(
    train_input=train_split['features'],
    train_target=train_split['targets'],
    train_states=train_split['states'],
    train_graph=adj,
    train_dynamic_graph=train_split['dynamic_graph'],

    val_input=val_split['features'],
    val_target=val_split['targets'],
    val_states=val_split['states'],
    val_graph=adj,
    val_dynamic_graph=val_split['dynamic_graph'],

    loss='mse',
    epochs=epochs,

    use_epi_reg=0.1,
    epi_reg_loss='mse',
    epi_hidden=8,
    epi_mode="sir_incidence",
    epi_percent_scale=100.0,
    epi_regions=data_df.columns.to_list(),
)
```

---

## Metrics

### Point metrics
- **MAE**: $\mathrm{MAE}=\frac{1}{n}\sum |\hat{y}-y|$
- **MSE**: $\mathrm{MSE}=\frac{1}{n}\sum (\hat{y}-y)^2$
- **RMSE**: $\mathrm{RMSE}=\sqrt{\mathrm{MSE}}$
- **medAE / medSE**: median absolute / squared error (robust to spikes)

### Filtered metrics
Filtered metrics (e.g., `get_MSE_filtered`) compute the same error after filtering targets to reduce the effect of heavy zeros/outliers (implementation follows IQR-style trimming + optional zero exclusion).

---

## Notes

The current `BaseModel` includes everything needed for:
- **TID** (via `_FutureTI` and `_apply_future_ti`)
- **Epi regularization** (via `_setup_epi_reg_from_data` and `_apply_epi_reg_loss`)
- **Probabilistic sampling + quantiles** (`predict_samples`, `predict_quantiles`)