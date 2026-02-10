from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


@dataclass
class Metrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    auc_macro_ovr: Optional[float]

    def to_dict(self) -> Dict[str, float]:
        d = {
            "accuracy": float(self.accuracy),
            "precision_macro": float(self.precision_macro),
            "recall_macro": float(self.recall_macro),
            "f1_macro": float(self.f1_macro),
        }
        if self.auc_macro_ovr is not None:
            d["auc_macro_ovr"] = float(self.auc_macro_ovr)
        return d


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Metrics:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            auc = None

    return Metrics(accuracy=acc, precision_macro=p, recall_macro=r, f1_macro=f1, auc_macro_ovr=auc)


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def make_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names=None) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
