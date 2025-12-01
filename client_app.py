import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from opacus import PrivacyEngine
from task import get_model, load_client_data, DEVICE

EPSILON = 50.0
MAX_GRAD_NORM = 1.2

class BrainTumorClient(NumPyClient):
    def __init__(self, train_loader):
        self.net = get_model()
        self.train_loader = train_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.train_dp()
        # Return loss AND accuracy to the server
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": float(loss), "accuracy": float(accuracy)}

    def train_dp(self):
        optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        privacy_engine = PrivacyEngine()
        self.net.train()
        
        model_dp, optimizer_dp, train_loader_dp = privacy_engine.make_private_with_epsilon(
            module=self.net,
            optimizer=optimizer,
            data_loader=self.train_loader,
            epochs=1,
            target_epsilon=EPSILON,
            target_delta=1e-5,
            max_grad_norm=MAX_GRAD_NORM,
        )

        correct = 0
        total = 0
        epoch_loss = 0.0
        n_batches = 0
        
        for mri, ct, labels in train_loader_dp:
            if len(labels) == 0: continue # Skip empty batches
            
            mri, ct, labels = mri.to(DEVICE), ct.to(DEVICE), labels.to(DEVICE)
            
            optimizer_dp.zero_grad()
            outputs = model_dp(mri, ct)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_dp.step()
            
            # Metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        avg_acc = correct / total if total > 0 else 0.0
        
        return avg_loss, avg_acc

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    train_loader = load_client_data(partition_id)
    return BrainTumorClient(train_loader).to_client()

app = ClientApp(client_fn=client_fn)