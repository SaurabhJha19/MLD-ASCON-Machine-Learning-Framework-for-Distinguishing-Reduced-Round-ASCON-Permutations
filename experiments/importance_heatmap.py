import numpy as np
import matplotlib.pyplot as plt

importance = np.array([
    [0.0,      0.0,      0.0,      0.0,      0.0],      
    [0.0,      0.0,      0.0,      0.0,      0.0],      
    [0.00075,  0.00145,  0.00175,  0.00105,  0.00160],
])

plt.figure(figsize=(7, 3))

im = plt.imshow(importance, cmap="viridis", aspect="auto")

plt.xticks(range(5), ["x0", "x1", "x2", "x3", "x4"])
plt.yticks(range(3), ["Round 1", "Round 2", "Round 3"])

plt.xlabel("ASCON Word")
plt.ylabel("Observed Intermediate Round")
plt.title("Word-Level Importance Across Intermediate Rounds")

plt.colorbar(im, label="Permutation Importance")

plt.tight_layout()

plt.savefig("results/importance_heatmap.png", dpi=300)
print("Saved to results/importance_heatmap.png")

plt.show()