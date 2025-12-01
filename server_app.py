import torch
import torch.nn as nn
import json
import os
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from task import get_model, load_global_val_data, load_global_test_data, DEVICE

# Helper function for centralized evaluation
def evaluate_global_model(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for mri, ct, labels in dataloader:
            mri, ct, labels = mri.to(DEVICE), ct.to(DEVICE), labels.to(DEVICE)
            outputs = model(mri, ct)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(dataloader), correct / total

class ResearchStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = []

    def aggregate_fit(self, server_round, results, failures):
        # 1. Aggregate Weights
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            print(f"Round {server_round}: Aggregating metrics...")
            
            # --- A. Load New Global Weights ---
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            model = get_model()
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            torch.save(model.state_dict(), "global_model.pth")
            
            # --- B. Calculate TRAIN Metrics (from Clients) ---
            train_losses = [r.metrics["loss"] for _, r in results]
            train_accs = [r.metrics["accuracy"] for _, r in results]
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_acc = sum(train_accs) / len(train_accs)
            
            # --- C. Calculate VAL & TEST Metrics (Centralized) ---
            val_loader = load_global_val_data()
            test_loader = load_global_test_data()
            
            val_loss, val_acc = evaluate_global_model(model, val_loader)
            test_loss, test_acc = evaluate_global_model(model, test_loader)
            
            print(f"  Train Acc: {avg_train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

            # --- D. Save History ---
            entry = {
                "round": server_round,
                "train_loss": avg_train_loss, "train_acc": avg_train_acc,
                "val_loss": val_loss,         "val_acc": val_acc,
                "test_loss": test_loss,       "test_acc": test_acc
            }
            self.history.append(entry)
            with open("history.json", "w") as f:
                json.dump(self.history, f)
            
        return aggregated_parameters, aggregated_metrics

def server_fn(context: Context):
    num_rounds = context.run_config.get("num-server-rounds", 3)
    initial_model = get_model()
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    )

    strategy = ResearchStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        initial_parameters=initial_params
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)