import matplotlib.pyplot as plt

rounds = [1, 2, 3, 4]
accuracy = [1.0000, 1.0000, 0.9995, 0.5005]

plt.figure(figsize=(6,4))
plt.plot(rounds, accuracy, marker='o', linewidth=2)
plt.xticks(rounds)
plt.ylim(0.45, 1.02)
plt.xlabel('Observed Intermediate Round')
plt.ylabel('MLP Accuracy')
plt.title('Intermediate-Round Differential Distinguishability in ASCON')
plt.grid(True)
plt.tight_layout()

plt.savefig('results/intermediate_decay.png', dpi=300)
print('Saved to results/intermediate_decay.png')

plt.show()