import os
import glob
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
import torch
from transformers import AutoModel
from sklearn.metrics import balanced_accuracy_score

# Local imports
from src.prompts_zeroshot import (
    brca_prompts, rcc_prompts, nsclc_prompts,
    esca_prompts, tgct_prompts, cesc_prompts
)

# =============================================================================
# Configuration
# =============================================================================
PATH_TEST_SLIDES = "./slide_feats_for_zeroshot/"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load TITAN model (pretrained on pathology foundation model)
titan_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
titan_model = titan_model.to(DEVICE)
titan_model.eval()

# =============================================================================
# Prompt Setup: Collect class prompts and templates
# =============================================================================
_, TEMPLATES = brca_prompts()  # Get templates from first prompt set (shared)
CLASS_PROMPTS = []

prompt_functions = [
    brca_prompts, rcc_prompts, nsclc_prompts,
    esca_prompts, tgct_prompts, cesc_prompts
]

for prompt_fn in prompt_functions:
    class_prompts, _ = prompt_fn()
    CLASS_PROMPTS.extend(class_prompts)

# Build zero-shot classifier once (shared across all tasks)
with torch.autocast('cuda', dtype=torch.float16), torch.inference_mode():
    classifier = titan_model.zero_shot_classifier(
        CLASS_PROMPTS, TEMPLATES, device=DEVICE
    )

# Define class index mapping per task: {task_id: [start_idx, end_idx]}
DICT_CLASSES = {
    0: [0, 1],   # BRCA
    1: [2, 4],   # RCC
    2: [5, 6],   # NSCLC
    3: [7, 8],   # ESCA
    4: [9, 10],  # TGCT
    5: [11, 12]  # CESC
}

NUM_TASKS = len(DICT_CLASSES)

# =============================================================================
# Utility Functions
# =============================================================================
def backward_transfer(results: List[List[float]]) -> float:
    """
    Compute Backward Transfer (BWT): average drop in performance on previous tasks
    after learning the last task.
    """
    n_tasks = len(results)
    bwt_scores = []
    for i in range(n_tasks - 1):
        bwt_scores.append(results[-1][i] - results[i][i])
    return float(np.mean(bwt_scores))


def forgetting(results: List[List[float]]) -> float:
    """
    Compute Forgetting: average drop from best past performance to final performance
    on previous tasks.
    """
    n_tasks = len(results)
    # Pad shorter sequences
    padded = [row + [0.0] * (n_tasks - len(row)) for row in results]
    np_res = np.array(padded)
    max_per_task = np.max(np_res, axis=0)  # Best performance per task
    forgetting_scores = []
    for i in range(n_tasks - 1):
        forgetting_scores.append(max_per_task[i] - results[-1][i])
    return float(np.mean(forgetting_scores))


# =============================================================================
# Main Evaluation Loop over Folds
# =============================================================================
metrics_all_folds = {
    "acc": [],           # Overall accuracy after all tasks
    "masked_acc": [],    # Masked accuracy (only current task classes)
    "mean_acc": [],      # Mean per-task accuracy at final step
    "bwt": [],           # Backward transfer
    "forgetting": []     # Forgetting measure
}

# Per-fold accumulators
bacc_all_folds = []
masked_bacc_all_folds = []
accumulated_accs_all_folds = []  # Accuracy as tasks are accumulated
acc_per_task_final = []          # Final per-task accuracy (unmasked)
masked_acc_per_task_final = []   # Final per-task masked accuracy

for fold in tqdm(range(10), desc="Folds"):
    fold_path = os.path.join(PATH_TEST_SLIDES, str(fold))
    
    # Load all slides and group by task
    tasks: Dict[int, List[Tuple[torch.Tensor, int, int]]] = {i: [] for i in range(NUM_TASKS)}
    
    for slide_path in sorted(glob.glob(os.path.join(fold_path, "**", "*.pt")) + 
                             glob.glob(os.path.join(fold_path, "**", "*.pth"))):
        slide = torch.load(slide_path, map_location=DEVICE)
        filename = os.path.basename(slide_path)
        label_str = filename.split("_label_")[-1].split('.')[0]
        task_id = int(slide_path.split('/')[-2].split('_')[-1])
        label = int(label_str)
        
        tasks[task_id].append((slide, label, task_id))

    # Accumulate accuracy over incremental task sequences
    acc_per_sequence = []           # List of [acc_task0, acc_task1, ...] at each step
    masked_acc_per_sequence = []    # Masked version
    accumulated_overall_accs = []   # Overall accuracy as more tasks are added

    for seq_len in range(1, NUM_TASKS + 1):  # seq_len = 1 to 6 tasks
        current_tasks = list(range(seq_len))
        test_slides = [item for t in current_tasks for item in tasks[t]]

        # Metrics storage
        correct_total = 0
        correct_masked_total = 0
        acc_per_task = [0.0] * seq_len
        masked_acc_per_task = [0.0] * seq_len

        # For balanced accuracy
        true_labels = {t: [] for t in current_tasks}
        pred_labels = {t: [] for t in current_tasks}
        true_labels_masked = {t: [] for t in current_tasks}
        pred_labels_masked = {t: [] for t in current_tasks}

        with torch.autocast('cuda', dtype=torch.float16), torch.inference_mode():
            for slide, label, task_id in test_slides:
                # === Unmasked (Full Classifier) ===
                end_idx = DICT_CLASSES[seq_len - 1][-1]  # Last class index in current seq
                scores = titan_model.zero_shot(slide, classifier[:, :end_idx + 1])
                pred_class = scores.argmax().item()

                correct_total += (pred_class == label)
                acc_per_task[task_id] += (pred_class == label)

                true_labels[task_id].append(label)
                pred_labels[task_id].append(pred_class)

                # === Masked (Only Current Task Classes) ===
                start_idx, end_idx_task = DICT_CLASSES[task_id]
                scores_masked = titan_model.zero_shot(slide, classifier[:, start_idx:end_idx_task + 1])
                pred_masked = scores_masked.argmax().item()
                true_label_masked = label - start_idx

                correct_masked_total += (pred_masked == true_label_masked)
                masked_acc_per_task[task_id] += (pred_masked == true_label_masked)

                true_labels_masked[task_id].append(true_label_masked)
                pred_labels_masked[task_id].append(pred_masked)

        # Normalize accuracies
        task_sizes = [len(tasks[t]) for t in current_tasks]
        acc_per_task = [acc_per_task[i] / task_sizes[i] for i in range(seq_len)]
        masked_acc_per_task = [masked_acc_per_task[i] / task_sizes[i] for i in range(seq_len)]

        # Store sequence-level results
        acc_per_sequence.append(acc_per_task)
        masked_acc_per_sequence.append(masked_acc_per_task)
        accumulated_overall_accs.append(correct_total / len(test_slides))

        # Save logits (optional, for analysis)
        if seq_len == NUM_TASKS:
            torch.save(true_labels, f"logits_seq_{seq_len-1}_labels.pth")
            torch.save(pred_labels, f"logits_seq_{seq_len-1}_preds.pth")

    # === Final Metrics for This Fold ===
    final_acc = accumulated_overall_accs[-1]
    final_masked_acc = correct_masked_total / sum(len(tasks[t]) for t in range(NUM_TASKS))

    # Balanced accuracy
    bacc_per_task = [
        balanced_accuracy_score(true_labels[t], pred_labels[t])
        for t in range(NUM_TASKS)
    ]
    masked_bacc_per_task = [
        balanced_accuracy_score(true_labels_masked[t], pred_labels_masked[t])
        for t in range(NUM_TASKS)
    ]

    # Store fold results
    metrics_all_folds["acc"].append(final_acc)
    metrics_all_folds["masked_acc"].append(final_masked_acc)
    metrics_all_folds["mean_acc"].append(np.mean(acc_per_sequence[-1]))
    metrics_all_folds["bwt"].append(backward_transfer(acc_per_sequence))
    metrics_all_folds["forgetting"].append(forgetting(acc_per_sequence))

    bacc_all_folds.append(np.mean(bacc_per_task))
    masked_bacc_all_folds.append(np.mean(masked_bacc_per_task))
    accumulated_accs_all_folds.append(accumulated_overall_accs)

    acc_per_task_final.append(acc_per_sequence[-1])
    masked_acc_per_task_final.append(masked_acc_per_sequence[-1])

    print(f"Fold {fold} - Masked ACC: {final_masked_acc:.4f}")

# =============================================================================
# Final Reporting
# =============================================================================
def mean_std(values, scale=1.0):
    return np.mean(values) * scale, np.std(values) * scale

print("\n=== Final Results (10-Fold) ===")
print(f"ACC:           {mean_std(metrics_all_folds['acc'], 100):.2f} ± {mean_std(metrics_all_folds['acc'], 100)[1]:.2f}")
print(f"MASKED_ACC:    {mean_std(metrics_all_folds['masked_acc'], 100):.2f} ± {mean_std(metrics_all_folds['masked_acc'], 100)[1]:.2f}")
print(f"bACC:          {mean_std(bacc_all_folds, 100):.2f} ± {mean_std(bacc_all_folds, 100)[1]:.2f}")
print(f"MASKED_bACC:   {mean_std(masked_bacc_all_folds, 100):.2f} ± {mean_std(masked_bacc_all_folds, 100)[1]:.2f}")
print(f"MEAN_ACC:      {mean_std(metrics_all_folds['mean_acc']):.4f} ± {mean_std(metrics_all_folds['mean_acc'])[1]:.4f}")
print(f"BWT:           {mean_std(metrics_all_folds['bwt'], 100):.2f} ± {mean_std(metrics_all_folds['bwt'], 100)[1]:.2f}")
print(f"FORGETTING:    {mean_std(metrics_all_folds['forgetting'], 100):.2f} ± {mean_std(metrics_all_folds['forgetting'], 100)[1]:.2f}")

# Per-task final accuracy
print("\nAcc per task (Final Step):")
acc_per_task = {tid: [] for tid in range(NUM_TASKS)}
for fold in range(10):
    for tid in range(NUM_TASKS):
        acc_per_task[tid].append(acc_per_task_final[fold][tid])

for tid in range(NUM_TASKS):
    m, s = np.mean(acc_per_task[tid]), np.std(acc_per_task_final[tid])
    print(f"Task {tid}: {m*100:5.2f} ± {s*100:4.2f}")

# Per-task final masked accuracy
print("\nAcc per task (Masked, Final Step):")
masked_acc_per_task = {tid: [] for tid in range(NUM_TASKS)}
for fold in range(10):
    for tid in range(NUM_TASKS):
        masked_acc_per_task[tid].append(masked_acc_per_task_final[fold][tid])

for tid in range(NUM_TASKS):
    m, s = np.mean(masked_acc_per_task[tid]), np.std(masked_acc_per_task[tid])
    print(f"Task {tid}: {m*100:5.2f} ± {s*100:4.2f}")

# Average accumulated accuracy curve
accumulated_accs_all_folds = np.array(accumulated_accs_all_folds)
print("\nAccumulated Accuracy (Mean ± Std):")
for i in range(NUM_TASKS):
    m, s = np.mean(accumulated_accs_all_folds[:, i]), np.std(accumulated_accs_all_folds[:, i])
    print(f"After {i+1} task(s): {m*100:5.2f} ± {s*100:4.2f}")