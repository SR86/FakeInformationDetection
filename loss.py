import json
import matplotlib.pyplot as plt
import sys

def plot_metrics_from_trainer_state(trainer_state_path):
    with open(trainer_state_path, 'r', encoding='utf-8') as f:
        trainer_state = json.load(f)

    log_history = trainer_state.get("log_history", [])
    if not log_history:
        print("No log history found.")
        return

    steps = []
    eval_loss = []
    eval_accuracy = []
    eval_f1 = []

    for entry in log_history:
        if 'eval_loss' in entry:
            steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])
            eval_accuracy.append(entry.get('eval_accuracy'))
            eval_f1.append(entry.get('eval_f1'))

    if not steps:
        print("No evaluation entries found in log history.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, eval_loss, marker='o', label="Eval Loss")
    plt.plot(steps, eval_accuracy, marker='x', label="Eval Accuracy")
    plt.plot(steps, eval_f1, marker='s', label="Eval F1")
    plt.xlabel("Training Steps")
    plt.ylabel("Metrics")
    plt.title("Evaluation Metrics Over Training Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <trainer_state.json path>")
    else:
        plot_metrics_from_trainer_state(sys.argv[1])
