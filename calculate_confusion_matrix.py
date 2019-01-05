'''
Calculate confusion matrix efficiently using torch.jit.script
ys: Labels
ys_: Predictions
cm: confusion matrix
    initial: torch.zeros((N, N), dtype=torch.int))
'''
import torch


@torch.jit.script
def calculate_confusion_matrix(ys, ys_, cm):
    ys, ys_ = ys.to(device='cpu'), ys_.to(device='cpu')

    for i in range(len(ys)):
        y, y_ = ys[i], ys_[i]
        cm[y][y_] += 1

    return cm
