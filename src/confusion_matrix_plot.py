import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm = np.array([
    [9444, 267],
    [4075, 8758]
])

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png")

print("Confusion Matrix graph saved successfully!")