import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



def main():
    try:
        with open("history.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Run 'flwr run .' first to generate history.json")
        return

    rounds = [d["round"] for d in data]
    train_loss = [d["train_loss"] for d in data]
    train_acc = [d["train_acc"] for d in data]
    val_acc = [d["val_acc"] for d in data]
    test_acc = [d["test_acc"] for d in data]


    # Calculate Gaps
    train_val_gap = np.array(train_acc) - np.array(val_acc)
    train_test_gap = np.array(train_acc) - np.array(test_acc)
    val_test_gap = np.abs(np.array(val_acc) - np.array(test_acc))

    # --- PLOT SETUP (2x2 Grid) ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Federated Learning Performance Metrics', fontsize=16)

    # 1. FL Training Loss
    axs[0, 0].plot(rounds, train_loss, 'o-', label='Train Loss')
    axs[0, 0].set_title('FL Training Loss')
    axs[0, 0].set_xlabel('Round')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True, alpha=0.3)

    # 2. FL Accuracies (Train vs Val vs Test)
    axs[0, 1].plot(rounds, train_acc, 'o-', label='Train')
    axs[0, 1].plot(rounds, val_acc, 's-', label='Val')
    axs[0, 1].plot(rounds, test_acc, '^-', label='Test')
    axs[0, 1].set_title('FL Accuracies')
    axs[0, 1].set_xlabel('Round')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Generalization Gaps
    axs[1, 0].plot(rounds, train_val_gap, 'o-', label='Train-Val Gap')
    axs[1, 0].plot(rounds, train_test_gap, 's-', label='Train-Test Gap')
    axs[1, 0].axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% Threshold')
    axs[1, 0].set_title('Generalization Gaps')
    axs[1, 0].set_xlabel('Round')
    axs[1, 0].set_ylabel('Accuracy Gap')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Val-Test Alignment
    axs[1, 1].plot(rounds, val_test_gap, 'o-', color='purple', label='|Val - Test| Gap')
    axs[1, 1].axhline(y=0.02, color='r', linestyle='--', alpha=0.5, label='2% Threshold')
    axs[1, 1].set_title('Val-Test Alignment')
    axs[1, 1].set_xlabel('Round')
    axs[1, 1].set_ylabel('Abs Gap')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("research_dashboard.png", dpi=300)
    print("✅ Created research_dashboard.png with all 4 graphs!")

if __name__ == "__main__":
    main()