"""Run Anomalib PatchCore and PaDiM on the same toy defect dataset.

Install requirements, then run generate_dataset.py before running this file.
Anomalib's validation subset is disjoint from its final test subset.
"""
from pathlib import Path
import csv

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score, confusion_matrix)

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


def collect(batches):
    """Extract image-level labels, scores and file names from Anomalib prediction batches."""
    names, y_true, y_pred, scores = [], [], [], []
    for batch in batches:
        if batch.gt_label is None or batch.pred_label is None or batch.pred_score is None:
            raise RuntimeError('Missing ground-truth labels or predicted scores/labels')
        paths = batch.image_path
        names.extend(str(p) for p in paths)
        y_true.extend(batch.gt_label.detach().cpu().numpy().astype(int).reshape(-1).tolist())
        y_pred.extend(batch.pred_label.detach().cpu().numpy().astype(int).reshape(-1).tolist())
        scores.extend(batch.pred_score.detach().cpu().numpy().reshape(-1).tolist())
    return names, np.array(y_true), np.array(y_pred), np.array(scores)


def save_anomaly_maps(name, batches):
    """Save one normal and one anomalous test image beside their predicted heatmaps."""
    examples = {}
    for batch in batches:
        if batch.anomaly_map is None or batch.gt_label is None:
            continue
        for i, truth in enumerate(batch.gt_label.detach().cpu().numpy().reshape(-1)):
            label = int(truth)
            if label not in examples:
                examples[label] = (
                    str(batch.image_path[i]),
                    batch.anomaly_map[i].detach().cpu().numpy().squeeze(),
                )
        if len(examples) == 2:
            break
    if len(examples) < 2:
        print('No normal+defect pair found for heatmap preview')
        return
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for row, label in enumerate([0, 1]):
        path, anomaly_map = examples[label]
        axes[row, 0].imshow(Image.open(path).convert('RGB'))
        axes[row, 0].set_title('Normal image' if label == 0 else 'Defect image')
        axes[row, 1].imshow(anomaly_map, cmap='magma')
        axes[row, 1].set_title(f'{name} anomaly heatmap')
        for ax in axes[row]:
            ax.axis('off')
    fig.tight_layout()
    filename=RESULTS / f'{name.lower()}_heatmaps.png'
    fig.savefig(filename, dpi=130)
    plt.close(fig)
    print(f'Saved {filename}')


def one_model(name, model):
    data = Folder(
        name='toy_concrete',
        root=DATA,
        normal_dir='train/good',
        normal_test_dir='test/good',
        abnormal_dir='test/defect',
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,       # more reliable on macOS interactive setups
        val_split_mode='from_test',
        val_split_ratio=0.5,
        seed=42,
    )
    engine = Engine(
        accelerator='cpu',  # avoids Apple MPS operator support issues
        devices=1,
        max_epochs=1,        # PatchCore / PaDiM fit their normal-feature models in one pass
        default_root_dir=str(RESULTS / name),
        logger=False,
        enable_progress_bar=False,
    )
    print(f'\n===== {name}: fitting on NORMAL images only =====')
    engine.fit(model=model, datamodule=data)
    print(f'===== {name}: evaluating on held-out test images =====')
    engine.test(model=model, datamodule=data)

    # Predict on ONLY the test_dataloader used for final evaluation,
    # not the data used to calibrate the model's decision threshold.
    batches = engine.predict(model=model,
                             dataloaders=data.test_dataloader(),
                             return_predictions=True)
    if not batches:
        raise RuntimeError('Anomalib returned no predictions')
    names, truth, pred, scores = collect(batches)
    if len(truth) == 0:
        raise RuntimeError('The held-out test split is empty')

    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0,1]).ravel()
    metrics = {
        'model': name,
        'evaluated_images': len(truth),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'accuracy': accuracy_score(truth,pred),
        'precision': precision_score(truth,pred,zero_division=0),
        'recall': recall_score(truth,pred,zero_division=0),
        'F1': f1_score(truth,pred,zero_division=0),
        'average_precision': average_precision_score(truth,scores)
            if len(np.unique(truth))>1 else float('nan'),
    }
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / f'{name.lower()}_predictions.csv').open('w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['image_path','true_label_defect_1','pred_label_defect_1','anomaly_score'])
        writer.writerows(zip(names, truth, pred, scores))
    save_anomaly_maps(name, batches)
    print('\n', metrics)
    return metrics


if __name__ == '__main__':
    if not (DATA / 'train/good').exists():
        raise SystemExit('First run: python generate_dataset.py')
    all_metrics=[]
    all_metrics.append(one_model('PatchCore', Patchcore(
        backbone='resnet18', layers=['layer2','layer3'],
        coreset_sampling_ratio=0.05, num_neighbors=1,
        visualizer=False,
    )))
    all_metrics.append(one_model('PaDiM', Padim(
        backbone='resnet18', layers=['layer1','layer2','layer3'],
        n_features=30,
        visualizer=False,
    )))
    with (RESULTS/'comparison.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    print('\nSaved comparison to results/comparison.csv')
    print('Try reviewing predictions CSV files to see missed defects and false alarms.')
