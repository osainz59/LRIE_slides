import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.interpolate import interp1d
import rich
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

x = np.asarray([0.0, 1.0, 5.0, 10.0, 20.0, 100.0])

PRIMARY_COLOR = "#08457E"
SECONDARY_COLOR = "#4b16fe"
TERTIARY_COLOR = "#009999"

data = {
    "Developer A": [
        np.asarray([40.3, 46.2, 56.3, 63.8, 69.6, 76.4]),
        np.asarray([0.00, 0.15, 0.97, 1.19, 0.53, 0.64]),
    ],
    "Developer B": [
        np.asarray([40.4, 44.9, 57.3, 64.1, 70.1, 73.3]),
        np.asarray([0.00, 0.53, 0.49, 0.66, 0.14, 0.94]),
    ],
}

fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(4, 3.6),
    layout="constrained",
)

ax.plot(
    x,
    data["Developer A"][0],
    label="Developer A",
    color=PRIMARY_COLOR,
    linestyle="-",
    marker="o",
    markersize=3.5,
)

ax.plot(
    x,
    data["Developer B"][0],
    label="Developer B",
    color=SECONDARY_COLOR,
    linestyle="-",
    marker="o",
    markersize=3.5,
)

ax.fill_between(
    x,
    data["Developer A"][0] - data["Developer A"][1],
    data["Developer A"][0] + data["Developer A"][1],
    color=PRIMARY_COLOR,
    alpha=0.2,
)

ax.fill_between(
    x,
    data["Developer B"][0] - data["Developer B"][1],
    data["Developer B"][0] + data["Developer B"][1],
    color=SECONDARY_COLOR,
    alpha=0.2,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)


# ax.set_ylim([0, 100])
ax.set_xscale("symlog")
ax.set_xlim([-0.2, 110])
ax.set_xticks(
    [0, 1, 5, 10, 20, 100], ["0%", "1%", "5%", "10%", "20%", "100%"]
)
ax.set_xlabel("Number of training examples (%)", fontsize=14)

fig.legend(loc="outside upper center", fontsize=14, ncol=2)
# fig.text(0.5, -0.06, "Number of training examples (%)", fontsize=14, ha="center")

# plt.show()

# plt.savefig("images/re_figure.png", dpi=400, bbox_inches='tight')
plt.savefig(
    f"images/developer_comaprison.pgf",
    bbox_inches="tight",
)

