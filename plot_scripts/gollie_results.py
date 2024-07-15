import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

plt.style.use("seaborn-v0_8-whitegrid")

x = np.asarray([0.0, 1.0, 5.0, 100.0])

PRIMARY_COLOR = "#08457E"
SECONDARY_COLOR = "#4b16fe"
TERTIARY_COLOR = "#009999"

results = {
    "Supervised": np.asarray([73.3, 73.0, 73.9, 75.0]),
    "Zero-Shot": np.asarray([42.3, 55.3, 56.0, 57.2]),
    "Seen": np.asarray([41.2, 54.6, 54.2, 54.7]),
    "Unseen": np.asarray([31.6, 47.5, 48.0, 51.4]),
}

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3.8), layout="constrained")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

for i, task in enumerate(["Supervised", "Zero-Shot"]):
    data = results[task]

    for j, model in enumerate(["Baseline 7B", "GoLLIE 7B", "GoLLIE 13B", "GoLLIE 34B"]):
        ax.bar(
            2*i + j*.3 -.45,
            data[j],
            color=PRIMARY_COLOR if j == 0 else SECONDARY_COLOR,
            label=model if not i else None,
            alpha=0.5 if j < 2 else 0.25,
            width=0.28,
            edgecolor=PRIMARY_COLOR if j == 0 else SECONDARY_COLOR,
            hatch=None if j < 2 else "/" if j == 2 else "\\",
        )

ax.set_xticks(
    [0, 2],
    labels=["\nSupervised", "\nZero-Shot"],
    fontsize=14,
)
# ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.75)
ax.grid(axis="x", alpha=0.0)
ax.set_ylabel("F1 Score", fontsize=14)

fig.legend(loc="outside upper center", fontsize=14, ncol=4)

plt.savefig("images/gollie_results.pgf", bbox_inches="tight")

##########################################################################################

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3.8), layout="constrained")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

for i, task in enumerate(["Seen", "Unseen"]):
    data = results[task]

    for j, model in enumerate(["Baseline 7B", "GoLLIE 7B", "GoLLIE 13B", "GoLLIE 34B"]):
        ax.bar(
            2*i + j*.3 -.45,
            data[j],
            color=PRIMARY_COLOR if j == 0 else SECONDARY_COLOR,
            label=model if not i else None,
            alpha=0.5 if j < 2 else 0.25,
            width=0.28,
            edgecolor=PRIMARY_COLOR if j == 0 else SECONDARY_COLOR,
            hatch=None if j < 2 else "/" if j == 2 else "\\",
        )

ax.set_xticks(
    [0, 2],
    labels=["\nSeen", "\nUnseen"],
    fontsize=14,
)
# ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.75)
ax.grid(axis="x", alpha=0.0)
ax.set_ylabel("F1 Score", fontsize=14)

fig.legend(loc="outside upper center", fontsize=14, ncol=4)

plt.savefig("images/gollie_see_unseen_results.pgf", bbox_inches="tight")



