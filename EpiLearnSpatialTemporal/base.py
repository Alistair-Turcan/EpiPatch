import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from .utils import *
from .metrics import get_loss

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

class BaseModel(nn.Module):
    def __init__(self, device = 'cpu', 
                 use_future_ti=False, tid_sizes=None, emb_dim=4, ti_hidden=(16,), node_specific=True, num_nodes=None):
        super(BaseModel, self).__init__()
        self.device = device
        self.future_ti = _FutureTI(tid_sizes, emb_dim, ti_hidden, node_specific, num_nodes).to(device) \
                         if (use_future_ti and tid_sizes) else None
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