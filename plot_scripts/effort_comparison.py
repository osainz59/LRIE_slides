import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# import tikzplotlib

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

x = np.asarray([0.0, 1.0, 5.0, 10.0, 20.0, 100]) * 180 / 100
print(x)

PRIMARY_COLOR = "#08457E"
SECONDARY_COLOR = "#4b16fe"
TERTIARY_COLOR = "#009999"

data = {
    "Data annotation": np.asarray([4.58, 37.5, 50.9, 58.7, 72.1]),
    "Verbalization": np.asarray([40.3, 46.2, 56.3, 63.8, 69.6, 76.4]),
    "Verbalization + Schema transfer": np.asarray([62.7, 64.2, 69.3, 71.6, 71.9, 74.9]),
}

fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(8, 4),
    layout="constrained",
)

ax.vlines(x=5, ymin=0, ymax=100, linestyles="dotted", colors="grey")

arr = matplotlib.patches.FancyArrowPatch(
    (4.75, 84), (0.25, 84), arrowstyle="->", mutation_scale=10, color="black", lw=1
)
ax.add_patch(arr)
ax.annotate("Verbalization", xy=(4.75, 90), ha="right", va="top", fontsize=11)

arr = matplotlib.patches.FancyArrowPatch(
    (5.25, 84), (9.5, 84), arrowstyle="->", mutation_scale=10, color="black", lw=1
)
ax.add_patch(arr)
ax.annotate("Annotation", xy=(5.25, 90), ha="left", va="top", fontsize=11)

ax.annotate("Data annotation (100%)", xy=(40, data["Data annotation"][-1] + 1), ha="right", va="bottom", fontsize=11)

ax.plot(
    x[1:],
    data["Data annotation"],
    label="Data annotation",
    color=PRIMARY_COLOR,
    linestyle="-",
    marker="o",
    markersize=5,
    markerfacecolor="white",
)
ax.plot(
    x,
    data["Data annotation"][-1] * np.ones_like(x),
    color=PRIMARY_COLOR,
    linestyle="--",
)

ax.plot(
    x + 5,
    data["Verbalization"],
    label="Verbalization",
    color=SECONDARY_COLOR,
    linestyle="-",
    marker="o",
    markersize=5,
    markerfacecolor="white",
)

ax.plot(
    x + 5,
    data["Verbalization + Schema transfer"],
    label="Verbalization + Schema transfer",
    color=TERTIARY_COLOR,
    linestyle="-",
    marker="o",
    markersize=5,
    markerfacecolor="white",
)

# ax.fill_between(
#     x[1:],
#     data["Data annotation"],
#     np.zeros_like(data["Data annotation"]),
#     color=PRIMARY_COLOR,
#     alpha=0.2,
# )

# ax.fill_between(
#     x,
#     data["Verbalization"],
#     data["Data annotation"],
#     color=SECONDARY_COLOR,
#     alpha=0.2,
# )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)


print_x = np.arange(0, 45, 10)
ax.set_ylim([0, 100])
# ax.set_xscale("symlog")
ax.set_xlim([print_x[0] - 0.5, print_x[-1]])
ax.set_xticks(
    np.arange(0, 45, 10),
    list(map("{:d}h".format, list(map(int, np.arange(0, 45, 10))))),
)
ax.set_xlabel("Hour of manual work", fontsize=14)
ax.set_ylabel("F1 Score", fontsize=14)

fig.legend(loc="outside upper center", fontsize=14, ncol=3)
# fig.text(0.5, -0.06, "Number of training examples (%)", fontsize=14, ha="center")

# plt.show()

# plt.savefig("images/re_figure.png", dpi=400, bbox_inches='tight')
plt.savefig(
    "images/effort_comparison.pgf",
    bbox_inches="tight",
)
