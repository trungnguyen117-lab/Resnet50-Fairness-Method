import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dữ liệu và nhãn
methods = ["US", "OS", "UW", "BM", "ADV", "DI", "BC+BB", "FLAC", "MFD", "FDR", "FR-B", "FR-P", "FAAP"]
metrics = np.array([
    [0.899, 0.890, 0.909, 0.938, 0.9, 0.921, 0.923, 0.942, 0.951, 0.860, 0.920, 0.930, 0.907],  # accuracy
    [0.924, 0.920, 0.935, 0.934, 0.923, 0.934, 0.929, 0.909, 0.923, 0.892, 0.916, 0.930, 0.770],  # balanced_accuracy
    [0.938, 0.934, 0.942, 0.950, 0.938, 0.946, 0.945, 0.947, 0.953, 0.922, 0.946, 0.948, 0.904],  # precision
    [0.914, 0.902, 0.918, 0.941, 0.914, 0.929, 0.930, 0.943, 0.948, 0.884, 0.938, 0.939, 0.905]   # f1_score
])
metric_labels = ["Accuracy", "Balanced Accuracy", "Precision", "F1 Score"]

# Tạo heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(metrics, annot=True, fmt=".3f", cmap="Greens", xticklabels=methods, yticklabels=metric_labels, cbar_kws={'label': 'Giá trị trung bình các chỉ số'})
plt.title("Bản đồ nhiệt trung bình các chỉ số", fontsize=16)
plt.xlabel("Phương pháp", fontsize=12)
plt.ylabel("Chỉ số", fontsize=12)
plt.tight_layout()
plt.show()
