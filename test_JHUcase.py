import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from EpiLearnSpatialTemporal.dataset import UniversalDataset
from EpiLearnSpatialTemporal import metrics
from EpiLearnSpatialTemporal.utils import generate_dataset
from EpiLearnSpatialTemporal.base import EinnModule

from EpiLearnSpatialTemporal.AGCRN import AGCRN
from EpiLearnSpatialTemporal.ColaGNN import ColaGNN
from EpiLearnSpatialTemporal.DCRNN import DCRNN
from EpiLearnSpatialTemporal.Dlinear import DlinearModel
from EpiLearnSpatialTemporal.EpiGNN import EpiGNN
from EpiLearnSpatialTemporal.EARTH import EARTH
from EpiLearnSpatialTemporal.GraphWaveNet import GraphWaveNet
from EpiLearnSpatialTemporal.MTGNN import MTGNN
from EpiLearnSpatialTemporal.STGCN import STGCN
from EpiLearnSpatialTemporal.GTS import GTS
from EpiLearnSpatialTemporal.StemGNN import StemGNN
from EpiLearnSpatialTemporal.STNorm import STNorm


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_splits(lookback=28, horizon=7, train_rate=0.6, val_rate=0.15, permute=False):
    data_df = pd.read_csv("rawData/processed/JHUcase.csv", index_col = 0)
    data_df.index = pd.to_datetime(data_df.index)
    adj_df = pd.read_csv("rawData/processed/JHUcase_adj.csv", index_col = 0)

    dataset = UniversalDataset()
    data = np.expand_dims(data_df.values, axis=-1)
    dataset.x = torch.FloatTensor(data)
    dataset.y = torch.FloatTensor(data)[:, :, 0]
    dataset.graph = torch.FloatTensor(adj_df.to_numpy())

    dow = torch.as_tensor(data_df.index.dayofweek.values, dtype=torch.long)
    woy = torch.as_tensor(data_df.index.isocalendar().week.values - 1, dtype=torch.long)
    dataset.states = torch.stack([woy], dim=-1)
    tid_s = {'dow': 7}

    train_dataset, val_dataset, test_dataset = dataset.ganerate_splits(
        train_rate=train_rate, val_rate=val_rate
    )

    train_input, train_target, _, train_states_future, train_adj = generate_dataset(
        X=train_dataset["features"],
        Y=train_dataset["target"],
        states=train_dataset["states"],
        dynamic_adj=train_dataset["dynamic_graph"],
        lookback_window_size=lookback,
        horizon_size=horizon,
        ahead=0,
        permute=permute,
    )
    val_input, val_target, _, val_states_future, val_adj = generate_dataset(
        X=val_dataset["features"],
        Y=val_dataset["target"],
        states=val_dataset["states"],
        dynamic_adj=val_dataset["dynamic_graph"],
        lookback_window_size=lookback,
        horizon_size=horizon,
        ahead=0,
        permute=permute,
    )
    test_input, test_target, _, test_states_future, test_adj = generate_dataset(
        X=test_dataset["features"],
        Y=test_dataset["target"],
        states=test_dataset["states"],
        dynamic_adj=test_dataset["dynamic_graph"],
        lookback_window_size=lookback,
        horizon_size=horizon,
        ahead=0,
        permute=permute,
    )

    splits = {
        "train": {
            "features": train_input,
            "targets": train_target,
            "states": train_states_future,
            "dynamic_graph": train_adj,
        },
        "val": {
            "features": val_input,
            "targets": val_target,
            "states": val_states_future,
            "dynamic_graph": val_adj,
        },
        "test": {
            "features": test_input,
            "targets": test_target,
            "states": test_states_future,
            "dynamic_graph": test_adj,
        },
    }

    return data_df, dataset.graph, splits, tid_s, train_dataset


def build_retraining_splits(
    lookback=28,
    horizon=7,
    min_train_size=None,
    val_size=None,
    test_size=None,
    step_size=None,
    permute=False,
):
    data_df = pd.read_csv("rawData/processed/JHUcase.csv", index_col=0)
    data_df.index = pd.to_datetime(data_df.index)
    adj_df = pd.read_csv("rawData/processed/JHUcase_adj.csv", index_col=0)

    dataset = UniversalDataset()
    data = np.expand_dims(data_df.values, axis=-1)
    dataset.x = torch.FloatTensor(data)
    dataset.y = torch.FloatTensor(data)[:, :, 0]
    dataset.graph = torch.FloatTensor(adj_df.to_numpy())

    woy = torch.as_tensor(data_df.index.isocalendar().week.values - 1, dtype=torch.long)
    dataset.states = torch.stack([woy], dim=-1)
    tid_s = {"woy": 53}

    transformed_dataset = dataset.get_transformed()
    adj_static = transformed_dataset["graph"] if transformed_dataset["graph"] is not None else dataset.graph
    total_steps = transformed_dataset["features"].shape[0]

    if min_train_size is None:
        min_train_size = lookback + horizon
    if val_size is None:
        val_size = horizon
    if test_size is None:
        test_size = horizon
    if step_size is None:
        step_size = test_size

    def slice_dataset(start, end):
        return {
            "features": transformed_dataset["features"][start:end],
            "target": transformed_dataset["target"][start:end],
            "graph": adj_static,
            "dynamic_graph": None
            if transformed_dataset["dynamic_graph"] is None
            else transformed_dataset["dynamic_graph"][start:end],
            "states": None
            if transformed_dataset["states"] is None
            else transformed_dataset["states"][start:end],
        }

    splits_list = []
    first_train_dataset = None
    for train_end in range(min_train_size, total_steps - val_size - test_size + 1, step_size):
        val_end = train_end + val_size
        test_end = val_end + test_size

        train_dataset = slice_dataset(0, train_end)
        val_dataset = slice_dataset(train_end, val_end)
        test_dataset = slice_dataset(val_end, test_end)
        if first_train_dataset is None:
            first_train_dataset = train_dataset

        train_input, train_target, _, train_states_future, train_adj = generate_dataset(
            X=train_dataset["features"],
            Y=train_dataset["target"],
            states=train_dataset["states"],
            dynamic_adj=train_dataset["dynamic_graph"],
            lookback_window_size=lookback,
            horizon_size=horizon,
            ahead=0,
            permute=permute,
        )
        val_input, val_target, _, val_states_future, val_adj = generate_dataset(
            X=val_dataset["features"],
            Y=val_dataset["target"],
            states=val_dataset["states"],
            dynamic_adj=val_dataset["dynamic_graph"],
            lookback_window_size=lookback,
            horizon_size=horizon,
            ahead=0,
            permute=permute,
        )
        test_input, test_target, _, test_states_future, test_adj = generate_dataset(
            X=test_dataset["features"],
            Y=test_dataset["target"],
            states=test_dataset["states"],
            dynamic_adj=test_dataset["dynamic_graph"],
            lookback_window_size=lookback,
            horizon_size=horizon,
            ahead=0,
            permute=permute,
        )

        if train_input.numel() == 0 or val_input.numel() == 0 or test_input.numel() == 0:
            continue

        splits_list.append(
            {
                "train": {
                    "features": train_input,
                    "targets": train_target,
                    "states": train_states_future,
                    "dynamic_graph": train_adj,
                },
                "val": {
                    "features": val_input,
                    "targets": val_target,
                    "states": val_states_future,
                    "dynamic_graph": val_adj,
                },
                "test": {
                    "features": test_input,
                    "targets": test_target,
                    "states": test_states_future,
                    "dynamic_graph": test_adj,
                },
                "test_sample_start": val_end,
            }
        )

    return data_df, adj_static, splits_list, tid_s, first_train_dataset, dataset


def compute_dtw_matrix(train_dataset, dataset_name, cache_dir="."):
    try:
        from fastdtw import fastdtw
    except ImportError as exc:
        raise ImportError("fastdtw is required to compute the DTW matrix.") from exc
    from tqdm import tqdm

    cache_path = os.path.join(cache_dir, f"dtw_{dataset_name}.npy")
    if os.path.exists(cache_path):
        dtw_matrix = np.load(cache_path)
        print(f"Loaded DTW matrix from {cache_path}")
        return dtw_matrix

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
    return dtw_matrix


def build_model(name, lookback, horizon, num_nodes, adj, tid_s, use_future_ti, device, dtw_matrix=None):
    common = dict(
        num_timesteps_input=lookback,
        num_timesteps_output=horizon,
        adj_m=adj,
        num_nodes=num_nodes,
        num_features=1,
        device=device,
        use_future_ti=use_future_ti,
        tid_sizes=tid_s,
        emb_dim=4,
        ti_hidden=(8,),
    )
    if name == "AGCRN":
        return AGCRN(rnn_units=16, nlayers=2, embed_dim=8, cheb_k=2, **common)
    if name == "ColaGNN":
        return ColaGNN(nhid=16, n_layer=2, **common)
    if name == "DCRNN":
        return DCRNN(max_diffusion_step=3, **common)
    if name == "EpiGNN":
        return EpiGNN(k=5, hidA=32, hidR=4, hidP=1, n_layer=2, dropout=0.2, **common)
    if name == "EARTH":
        if dtw_matrix is None:
            raise ValueError("EARTH requires dtw_matrix.")
        return EARTH(
            dtw_matrix=dtw_matrix,
            dropout=0.2,
            n_hidden=16,
            **common,
        )
    if name == "GraphWaveNet":
        return GraphWaveNet(
            residual_channels=4,
            dilation_channels=4,
            skip_channels=32,
            end_channels=64,
            kernel_size=2,
            blocks=4,
            nlayers=8,
            **common,
        )
    if name == "MTGNN":
        return MTGNN(
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
            **common,
        )
    if name == "STGCN":
        return STGCN(nhids=16, **common)
    if name == "GTS":
        return GTS(rnn_units=32, max_diffusion_step=2, **common)
    if name == "StemGNN":
        return StemGNN(stack_cnt=2, multi_layer=4, dropout_rate=0.2, leaky_rate=0.2, **common)
    if name == "STNorm":
        return STNorm(channels=8, kernel_size=2, blocks=8, layers=2, **common)
    if name == "Dlinear":
        return DlinearModel(
            num_timesteps_input=lookback,
            num_timesteps_output=horizon,
            num_features=1,
            num_nodes=num_nodes,
            use_future_ti=use_future_ti,
            tid_sizes=tid_s,
            emb_dim=4,
            ti_hidden=(8,),
            device=device,
        )
    raise ValueError(f"Unknown model name: {name}")


def eval_metrics(pred, target):
    mse = metrics.get_MSE(pred, target)
    mae = metrics.get_MAE(pred, target)
    rmse = metrics.get_RMSE(pred, target)
    mse_filtered = metrics.get_MSE_filtered(pred, target)
    mae_filtered = metrics.get_MAE_filtered(pred, target)
    medse = metrics.get_medSE(pred, target)
    medae = metrics.get_medAE(pred, target)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mse_filtered": mse_filtered,
        "mae_filtered": mae_filtered,
        "medse": medse,
        "medae": medae,
    }


def eval_horizon_metrics(pred, target, prefix="h"):
    horizon = target.shape[1]
    outputs = {}
    for h in range(horizon):
        metrics_out = eval_metrics(pred[:, h, :], target[:, h, :])
        for key, value in metrics_out.items():
            outputs[f"{prefix}{h + 1}_{key}"] = value
    return outputs


def aggregate_metrics(metrics_list):
    summary = {}
    if not metrics_list:
        return summary
    keys = metrics_list[0].keys()
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        tensors = [v if torch.is_tensor(v) else torch.tensor(v) for v in values]
        summary[key] = torch.stack(tensors).mean()
    return summary


def run_experiment(
    model_name,
    splits,
    adj,
    tid_s,
    use_future_ti,
    epi_mode,
    use_einn,
    loss_name,
    horizon,
    device,
    dtw_matrix=None,
    epochs=100,
):
    model = build_model(
        model_name,
        lookback=splits["train"]["features"].shape[1],
        horizon=horizon,
        num_nodes=adj.shape[0],
        adj=adj,
        tid_s=tid_s,
        use_future_ti=use_future_ti,
        device=device,
        dtw_matrix=dtw_matrix,
    )

    model.fit(
        train_input=splits["train"]["features"],
        train_target=splits["train"]["targets"],
        train_states=splits["train"]["states"],
        train_graph=adj,
        train_dynamic_graph=splits["train"]["dynamic_graph"],
        val_input=splits["val"]["features"],
        val_target=splits["val"]["targets"],
        val_states=splits["val"]["states"],
        val_graph=adj,
        val_dynamic_graph=splits["val"]["dynamic_graph"],
        loss=loss_name,
        epochs=epochs,
        use_epi_reg=False if not epi_mode else 0.1,
        epi_mode=epi_mode,
    )

    if use_einn and epi_mode:
        einn = EinnModule(
            num_nodes=adj.shape[0],
            horizon=horizon,
            in_features=splits["train"]["features"].shape[-1],
            epi_mode=epi_mode,
        ).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(einn.parameters()), lr=1e-3
        )
        model.train()
        einn.train()
        y_hat = model(
            splits["train"]["features"],
            adj,
            splits["train"]["states"],
            splits["train"]["dynamic_graph"],
        )
        L_base = F.mse_loss(y_hat, splits["train"]["targets"])
        L_ode, L_data, y_einn = einn.losses(
            splits["train"]["features"],
            splits["train"]["targets"],
            graph=adj,
            dynamic_graph=splits["train"]["dynamic_graph"],
        )
        L_align = F.mse_loss(y_hat, y_einn)
        loss = L_base + 0.1 * L_ode + 0.1 * L_data + 0.1 * L_align
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        preds = model.predict(
            splits["test"]["features"],
            graph=adj,
            states=splits["test"]["states"],
            dynamic_graph=splits["test"]["dynamic_graph"],
        )

    targets = splits["test"]["targets"]
    out = eval_metrics(preds, targets)
    out.update(eval_horizon_metrics(preds, targets))

    model._fit_conformal(
        splits["val"]["features"],
        splits["val"]["targets"],
        states=splits["val"]["states"],
        graph=adj,
        dynamic_graph=splits["val"]["dynamic_graph"],
    )
    crps_wis = model.compute_crps_wis(
        splits["test"]["features"],
        targets,
        quantile_levels=(0.5, 0.05, 0.95, 0.10, 0.90, 0.15, 0.85),
        alphas=(0.10, 0.20, 0.30),
        graph=adj,
        states=splits["test"]["states"],
        dynamic_graph=splits["test"]["dynamic_graph"],
        n_samples=100,
    )
    out.update(crps_wis)
    return out


def save_metrics(metrics_out, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    data = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics_out.items()}
    data["tag"] = tag
    path = os.path.join(out_dir, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(pd.Series(data).to_json())
    return data


def run_repeat_last_baseline(split, horizon):
    test_input = split["test"]["features"]
    targets = split["test"]["targets"]
    pred = test_input[:, -1, :, 0].unsqueeze(1).repeat(1, horizon, 1)
    out = eval_metrics(pred, targets)
    out.update(eval_horizon_metrics(pred, targets))
    return out


def run_today_shift_baseline(split, extended_target, horizon):
    targets = split["test"]["targets"]
    start_idx = split["test_sample_start"]
    outputs = []
    for fw_days in range(1, horizon + 1):
        shifted_start = start_idx - fw_days
        shifted_end = shifted_start + targets.shape[0]
        if shifted_start < 0:
            continue
        pred = extended_target[shifted_start:shifted_end]
        out = eval_metrics(pred, targets)
        out.update(eval_horizon_metrics(pred, targets))
        out["fw_days"] = fw_days
        outputs.append(out)
    return outputs


def run_arima_baseline(split, horizon, device):
    from EpiLearnSpatialTemporal.ARIMA import ARIMA

    model = ARIMA(
        num_timesteps_input=split["train"]["features"].shape[1],
        num_timesteps_output=horizon,
        num_features=1,
        device=device,
    )
    model.fit(
        train_input=split["train"]["features"],
        train_target=split["train"]["targets"],
    )
    preds = model.predict(split["test"]["features"])
    targets = split["test"]["targets"]
    out = eval_metrics(preds, targets)
    out.update(eval_horizon_metrics(preds, targets))
    return out


def run_retraining_experiment(
    model_name,
    splits_list,
    adj,
    tid_s,
    use_future_ti,
    epi_mode,
    use_einn,
    loss_name,
    horizon,
    device,
    dtw_matrix=None,
    epochs=100,
):
    metrics_list = []
    for split in splits_list:
        metrics_list.append(
            run_experiment(
                model_name=model_name,
                splits=split,
                adj=adj,
                tid_s=tid_s,
                use_future_ti=use_future_ti,
                epi_mode=epi_mode,
                use_einn=use_einn,
                loss_name=loss_name,
                horizon=horizon,
                device=device,
                dtw_matrix=dtw_matrix,
                epochs=epochs,
            )
        )
    return aggregate_metrics(metrics_list)

def main():
    dataset_name="JHUcase"
    fix_seed(42)
    device = "cpu"
    lookback = 28
    horizon = 7
    data_df, adj, splits_list, tid_s, train_dataset, dataset = build_retraining_splits(
        lookback=lookback,
        horizon=horizon,
    )
    if not splits_list:
        raise ValueError("No retraining splits generated; adjust window sizes.")
    adj = adj.type(torch.float)
    dtw_matrix = compute_dtw_matrix(train_dataset, dataset_name=dataset_name)
    _, extended_target, _, _, _ = generate_dataset(
        X=dataset.x,
        Y=dataset.y,
        lookback_window_size=lookback,
        horizon_size=horizon,
        ahead=0,
        permute=False,
    )
    out_dir = f"outputs_{dataset_name}"
    results = []

    model_names = [
        "Dlinear",
        "AGCRN",
        "ColaGNN",
        "DCRNN",
        "EpiGNN",
        "GraphWaveNet",
        "MTGNN",
        "STGCN",
        "GTS",
        "StemGNN",
        "STNorm",
        "EARTH",
    ]
    epi_modes = [False, "sir_incidence", "ngm"]
    loss_names = ["mse", "mse_filtered"]

    for model_name in model_names:
        for epi_mode in epi_modes:
            for loss_name in loss_names:
                use_filtering = loss_name == "mse_filtered"
                for use_einn in (False, True):
                    if use_einn and not epi_mode:
                        continue
                    for use_future_ti in (True, False):
                        tag = (
                            f"{model_name}|epi={epi_mode}|einn={use_einn}|filter={use_filtering}|ti={use_future_ti}"
                        )
                        metrics_out = run_retraining_experiment(
                            model_name=model_name,
                            splits_list=splits_list,
                            adj=adj,
                            tid_s=tid_s,
                            use_future_ti=use_future_ti,
                            epi_mode=epi_mode,
                            use_einn=use_einn,
                            loss_name=loss_name,
                            horizon=horizon,
                            device=device,
                            dtw_matrix=dtw_matrix if model_name == "EARTH" else None,
                        )
                        results.append(save_metrics(metrics_out, out_dir, tag))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, f"metrics_{dataset_name}.csv"), index=False)

    baseline_results = []
    repeat_metrics = aggregate_metrics(
        [run_repeat_last_baseline(split, horizon) for split in splits_list]
    )
    baseline_results.append(save_metrics(repeat_metrics, out_dir, "baseline_repeat_last"))

    today_metrics = []
    for split in splits_list:
        today_metrics.extend(run_today_shift_baseline(split, extended_target, horizon))
    for fw_days in range(1, horizon + 1):
        fw_metrics = [m for m in today_metrics if m.get("fw_days") == fw_days]
        fw_summary = aggregate_metrics(fw_metrics)
        baseline_results.append(save_metrics(fw_summary, out_dir, f"baseline_today_fw{fw_days}"))

    arima_metrics = aggregate_metrics(
        [run_arima_baseline(split, horizon, device) for split in splits_list]
    )
    baseline_results.append(save_metrics(arima_metrics, out_dir, "baseline_arima"))

if __name__ == "__main__":
    main()
