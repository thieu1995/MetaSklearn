
import numpy as np
from permetrics import ClassificationMetric



# Multi-class classification (three classes)
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])
y_pred = np.array([[0.9, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.1, 0.3, 0.6]])

cm = ClassificationMetric(y_true=y_true, y_pred=y_pred)
print(cm.crossentropy_loss())
