import sys

import numpy as np
from numpy.typing import ArrayLike

# Relative weights of the PLAsTiCC weighted log loss, keyed by the class names used in
# `snmachine.analysis.dict_label_to_real_plasticc`. PLAsTiCC gave double weight to
# KN and TDE (and to the anomalous class 99, which has no ELAsTiCC equivalent).
PLASTICC_CLASS_WEIGHTS: dict[str, float] = {
    "SNIa": 1,
    "SNIbc": 1,
    "SNII": 1,
    "SNIax": 1,
    "SNIa-91bg": 1,
    "SLSN-I": 1,
    "KN": 2,
    "TDE": 2,
    "AGN": 1,
}
DEFAULT_CLASS_WEIGHT: float = 1


def elasticc_log_loss(
    y_true: ArrayLike, probs: np.ndarray, weights: dict[str, float] | None = None
) -> float:
    """Weighted log loss for ELAsTiCC.

    No official ELAsTiCC metric was published, so this extends the PLAsTiCC
    weighted log loss: classes with a PLAsTiCC counterpart (matched by name, see
    `PLASTICC_CLASS_WEIGHTS`) keep their PLAsTiCC weight and every other class
    gets the same weight, `DEFAULT_CLASS_WEIGHT`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    probs : np.array of shape (n_samples, n_classes)
        Class probabilities for each sample. The order of the classes must be
        the same as `np.unique(y_true)`, which is the order of the attribute
        `classes_` of the classifier used.
    weights : dict, optional
        Relative weight of each class label. These override the default
        weights; labels not included use the defaults.

    Returns
    -------
    float
        Weighted log loss.
    """
    y_true = np.asarray(y_true)
    labels = np.unique(y_true)
    weights_dict = PLASTICC_CLASS_WEIGHTS | (weights or {})

    # Sanitize predictions; epsilon prevents log(0)
    epsilon = sys.float_info.epsilon
    probs = np.clip(probs, epsilon, 1.0 - epsilon)
    probs = probs / np.sum(probs, axis=1)[:, np.newaxis]
    log_probs = np.log(probs)

    class_logloss = [
        np.mean(log_probs[y_true == label, i]) for i, label in enumerate(labels)
    ]
    class_weights = [weights_dict.get(label, DEFAULT_CLASS_WEIGHT) for label in labels]
    return -1 * np.average(class_logloss, weights=class_weights)
