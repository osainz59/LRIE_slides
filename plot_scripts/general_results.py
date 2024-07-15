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
    "TACRED": {  # 0.0, 0.01, 0.05, 0.10, 0.20, 1.00
        "baseline": np.asarray([0.0, 07.7, 41.8, 55.1, None, 71.3]),
        "mnli": np.asarray([45.7, 56.1, 64.1, 67.8, None, 71.0]),
        "entailment": np.asarray([49.8, None, 70.5, None, None, 73.2]),
        "entailment+": None,
    },
    "ACE05": {  # 0.0, 0.01, 0.05, 0.10, 0.20, 1.00
        "baseline": np.asarray([0.0, 4.58, 37.5, 50.9, 58.7, 72.1]),
        "mnli": np.asarray([31.4, None, 46.0, None, None, 62.8]),
        "entailment": np.asarray([40.6, 45.4, 57.1, 64.6, 69.8, 74.6]),
        "entailment+": np.asarray([62.7, 64.2, 69.3, 71.6, 71.9, 74.9]),
    },
    "WikiEvent": {  # 0.0, 0.01, 0.05, 0.10, 0.20, 1.00
        "baseline": np.asarray([0.0, 16.9, 41.5, 49.9, 54.9, 61.3]),
        "mnli": np.asarray([29.5, None, 49.3, None, None, 59.9]),
        "entailment": np.asarray([35.9, 42.6, 52.2, 59.5, 65.4, 69.9]),
        "entailment+": np.asarray([57.3, 58.1, 65.2, 67.5, 67.9, 71.5]),
    },
}


def remove_none_points(x, y, z=None):
    x_ = np.copy(x) if x is not None else None
    y_ = np.copy(y) if y is not None else None
    z_ = np.copy(z) if z is not None else None
    if y is None:
        return x_, None, None
    if z is not None:
        # x_ = x[np.logical_and(z != None, y != None)].astype(float)
        # y_ = np.interp(x_, x_[y_ != None].astype(float), y_[y_ != None].astype(float))
        y_ = interp1d(
            np.log10(x_[y_ != None].astype(float) + 1), y_[y_ != None].astype(float)
        )(np.log10(x_ + 1))
        # y_ = y[np.logical_and(z != None, y != None)].astype(float)
        # z_ = np.interp(x_, x_[z_ != None].astype(float), z_[z_ != None].astype(float))
        z_ = interp1d(
            np.log(x_[z_ != None].astype(float) + 1), z_[z_ != None].astype(float)
        )(np.log(x_ + 1))
        # z_ = z[np.logical_and(z != None, y != None)].astype(float)
    else:
        x_ = x[y != None].astype(float)
        y_ = y[y != None].astype(float)
        z_ = None
    return x_, y_, z_


def plot(
    baseline: False,
    mnli: False,
    entailment: False,
    entailment_plus: False,
    TACRED=True,
    ACE05=True,
    WikiEvent=True,
):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=TACRED + ACE05 + WikiEvent,
        figsize=(4 * (TACRED + ACE05 + WikiEvent), 3.6),
        layout="constrained",
    )
    datasets = np.asarray(list(data.keys()))[np.asarray([TACRED, ACE05, WikiEvent])]
    for i, dataset in enumerate(datasets):
        values = data[dataset]

        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)

        baseline_x, baseline_y, _ = remove_none_points(x, values["baseline"])
        mnli_x, mnli_y, _ = remove_none_points(x, values["mnli"])
        entailment_x, entailment_y, _ = remove_none_points(x, values["entailment"])
        entailment_plus_x, entailment_plus_y, _ = remove_none_points(
            x, values["entailment+"]
        )

        if baseline:
            ax[i].plot(
                baseline_x,
                baseline_y,
                label="Supervised" if i == 1 else None,
                marker="o",
                color=PRIMARY_COLOR,  # "indigo",
                markersize=3.5,
            )
        if mnli:
            ax[i].plot(
                mnli_x,
                mnli_y,
                "--",
                label=r"Entailment\textsubscript{MNLI-only}" if i == 1 else None,
                marker="o",
                color=SECONDARY_COLOR,  # "goldenrod
                markersize=3.5,
            )
            ax[i].fill_between(
                mnli_x,
                mnli_y,
                np.zeros_like(mnli_y),
                color=SECONDARY_COLOR,  # "goldenrod",
                alpha=0.1,
            )
        if entailment:
            ax[i].plot(
                entailment_x,
                entailment_y,
                label="Entailment" if i == 1 else None,
                marker="o",
                color=SECONDARY_COLOR,  # "goldenrod",
                markersize=3.5,
            )

        if baseline:
            ax[i].fill_between(
                baseline_x,
                baseline_y,
                np.zeros_like(baseline_y),
                color=PRIMARY_COLOR,  # "goldenrod",
                alpha=0.1,
            )

        if baseline and entailment:
            baseline_x, baseline_y, entailment_y = remove_none_points(
                x, values["baseline"], values["entailment"]
            )
            ax[i].fill_between(
                baseline_x,
                baseline_y,
                entailment_y,
                color=SECONDARY_COLOR,  # "goldenrod",
                alpha=0.1,
            )

        if mnli and entailment:
            mnli_x, mnli_y, entailment_y = remove_none_points(
                x, values["mnli"], values["entailment"]
            )
            ax[i].fill_between(
                mnli_x,
                mnli_y,
                entailment_y,
                color=SECONDARY_COLOR,  # "goldenrod",
                alpha=0.1,
                hatch="\\\\",
            )

        if entailment_plus_y is not None and entailment_plus:
            ax[i].plot(
                entailment_plus_x,
                entailment_plus_y,
                label="Entailment + Schema Transfer" if i == 1 else None,
                marker="o",
                color=TERTIARY_COLOR,  # "goldenrod",
                markersize=3.5,
            )
            entailment_x, entailment_y, entailment_plus_y = remove_none_points(
                x, values["entailment"], values["entailment+"]
            )
            ax[i].fill_between(
                entailment_x,
                entailment_y,
                entailment_plus_y,
                color=TERTIARY_COLOR,  # "goldenrod",
                alpha=0.1,
            )

        ax[i].set_title(f"\n{dataset}", fontsize=16)
        # if i == 1:
        #     ax[i].set_xlabel(f"Number of training examples (%)", fontsize=14)
        # ax[i].set_xlabel(f"Number of training examples (%)", fontsize=14)

        if i == 0:
            ax[i].set_ylabel("F1 score", fontsize=16)

        ax[i].set_ylim([0, 100])
        ax[i].set_xscale("symlog")
        ax[i].set_xlim([-0.2, 110])
        ax[i].set_xticks(
            [0, 1, 5, 10, 20, 100], ["0%", "1%", "5%", "10%", "20%", "100%"]
        )

    fig.legend(loc="outside upper center", fontsize=16, ncol=4)
    fig.text(0.5, -0.06, "Number of training examples (%)", fontsize=16, ha="center")

    # plt.show()

    # plt.savefig("images/re_figure.png", dpi=400, bbox_inches='tight')
    plt.savefig(
        f"images/baseline={baseline}_mnli={mnli}_entailment={entailment}_plus={entailment_plus}.pgf",
        bbox_inches="tight",
    )


# tikzplotlib.save("images/re_figure.tex")
plot(baseline=True, mnli=False, entailment=True, entailment_plus=True, TACRED=False)
