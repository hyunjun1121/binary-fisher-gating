# arXiv Submission Information

## 1. Title

Binary Fisher Gating: Compressing Elastic Weight Consolidation for Efficient Continual Learning

## 2. Author(s)

Hyunjun Kim

## 3. Abstract

Continual learning methods face a fundamental tension between stability (retaining prior knowledge) and plasticity (learning new tasks). Existing approaches either store continuous importance weights per parameter (EWC, SI) or maintain per-task binary masks (PackNet, WSN)---incurring O(1) or O(T) storage respectively for T tasks.

We present Binary Fisher Gating (BFG), a simple yet effective parameter isolation method that achieves O(1) storage with a single cumulative 1-bit mask. BFG identifies the top-k% most important weights using Fisher Information after each task and permanently locks them via hard gradient masking. This closed-form approach requires no hyperparameter tuning beyond the lock fraction k, making it significantly simpler than learned mask methods (HAT, PackNet).

On Split-CIFAR-100 (10 tasks, 50 epochs/task), BFG achieves 68.8% accuracy with only 3.6% forgetting---dramatically outperforming EWC (49.3%, 35.2% forgetting) and SPG (41.3%, 46.8% forgetting). This 10x reduction in forgetting demonstrates that hard binary gating provides fundamentally stronger protection than soft regularization: locked weights experience zero drift, eliminating forgetting at its source. On Split-TinyImageNet, BFG maintains this advantage (38.8% vs. 24.7% for EWC). Requiring only 1 bit per weight (64x less than EWC), BFG is ideally suited for privacy-compliant edge deployment under strict storage constraints.

---

## Formatting Notes

- **Title**: No uppercase, no unicode, TeX macros expanded
- **Author**: First name first format
- **Abstract**: 1,429 characters (limit: 1,920), no font commands, TeX-isms removed
