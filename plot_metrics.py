import json
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Load Data
    try:
        with open("history.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: 'history.json' not found. Did you run the updated server_app.py?")
        return

    rounds = [entry["round"] for entry in data]
    loss = [entry["loss"] for entry in data]

    # 2. Plot Setup
    sns.set_theme(style="whitegrid") # Makes it look like a scientific paper
    plt.figure(figsize=(10, 6))
    
    # 3. Draw Line
    plt.plot(rounds, loss, marker='o', linestyle='-', color='#d62728', linewidth=2, label='Global Training Loss')
    
    # 4. Labels
    plt.title("Federated Model Convergence (Differential Privacy $\epsilon=50$)", fontsize=16, fontweight='bold')
    plt.xlabel("Federated Rounds", fontsize=14)
    plt.ylabel("Cross-Entropy Loss", fontsize=14)
    plt.xticks(rounds) # Ensure every round number is shown
    plt.legend()
    
    # 5. Save
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=300)
    print("✅ Saved training_curve.png")

if __name__ == "__main__":
    main()