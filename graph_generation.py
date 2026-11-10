import numpy as np
import matplotlib.pyplot as plt

# 示例数据：替换成你自己的训练 step 和 accuracy 即可
steps = np.arange(200, 8600, 200)

r1_zero_pass1 = [
    0.16, 0.23, 0.25, 0.29, 0.32, 0.37, 0.41, 0.38, 0.38, 0.38,
    0.49, 0.52, 0.52, 0.51, 0.58, 0.59, 0.60, 0.60, 0.56, 0.58,
    0.57, 0.54, 0.63, 0.58, 0.59, 0.62, 0.65, 0.63, 0.69, 0.67,
    0.66, 0.69, 0.65, 0.68, 0.67, 0.67, 0.69, 0.68, 0.71, 0.72,
    0.71, 0.71
]

r1_zero_pass16 = [
    0.25, 0.34, 0.31, 0.35, 0.45, 0.57, 0.55, 0.49, 0.55, 0.61,
    0.73, 0.69, 0.77, 0.77, 0.74, 0.77, 0.70, 0.74, 0.71, 0.78,
    0.73, 0.80, 0.80, 0.78, 0.80, 0.87, 0.77, 0.84, 0.80, 0.84,
    0.82, 0.81, 0.83, 0.84, 0.87, 0.86, 0.85, 0.83, 0.84, 0.86,
    0.87, 0.88
]

# baseline
o1_0912_pass1 = 0.745
o1_0912_cons64 = 0.835

plt.figure(figsize=(8, 4.8))

plt.plot(
    steps, r1_zero_pass1,
    color="blue", marker="o", markersize=5,
    linewidth=1.5, label="r1-zero-pass@1"
)

plt.plot(
    steps, r1_zero_pass16,
    color="red", marker="o", markersize=5,
    linewidth=1.5, label="r1-zero-cons@16"
)

plt.axhline(
    y=o1_0912_pass1,
    color="green", linestyle="--",
    linewidth=1.4, label="o1-0912-pass@1"
)

plt.axhline(
    y=o1_0912_cons64,
    color="magenta", linestyle="--",
    linewidth=1.4, label="o1-0912-cons@64"
)

plt.title("This is title")
plt.xlabel("Steps")
plt.ylabel("Accuracy")

plt.xlim(-200, 8800)
plt.ylim(0.12, 0.90)

plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("aime_accuracy.png", dpi=300)
plt.show()