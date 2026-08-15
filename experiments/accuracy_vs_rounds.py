import matplotlib.pyplot as plt

rounds = [2, 3, 4, 5]
accuracy = [1.0000, 0.9998, 0.5250, 0.4945]

plt.figure(figsize=(6, 4))
plt.plot(rounds, accuracy, marker="o", linewidth=2)
plt.xticks(rounds)
plt.ylim(0.45, 1.02)
plt.xlabel("ASCON Rounds")
plt.ylabel("MLP Accuracy")
plt.title("MLP Differential Distinguisher Accuracy vs ASCON Rounds")
plt.grid(True)
plt.tight_layout()

plt.savefig("results/accuracy_vs_rounds.png", dpi=300)
print("Saved to results/accuracy_vs_rounds.png")

plt.show()