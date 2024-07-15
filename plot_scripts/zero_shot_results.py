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

BAR_WIDTH = 0.5
SPACING = 0.1
BLOCK_SPACING = 2 * (2.5 * BAR_WIDTH + SPACING)
SHIFT = 3

results = {
    "NER": {
        "RoBERTa": {"Prec": 53.3, "Rec": 54.5, "F1": 53.9},
        'RoBERTa*': {'Prec': 73.5, 'Rec': 76.3, 'F1': 74.9},
        "DeBERTa": {"Prec": 58.0, "Rec": 50.2, "F1": 53.8},
        "RoBERTa+": {"Prec": 49.3, "Rec": 61.8, "F1": 54.9},
        'RoBERTa*+': {'Prec': 71.9, 'Rec': 77.8, 'F1': 74.8},
        "DeBERTa+": {"Prec": 56.3, "Rec": 63.1, "F1": 59.5},
        "Supervised": {"F1": 93.0},
    },
    "RE": {
        "RoBERTa": {"Prec": 32.8, "Rec": 75.5, "F1": 45.7},
        'RoBERTa*': {'Prec': 36.8, 'Rec': 76.7, 'F1': 49.8},
        "DeBERTa": {"Prec": 40.3, "Rec": 77.7, "F1": 53.0},
        "RoBERTa+": {"Prec": 56.1, "Rec": 55.8, "F1": 55.9},
        'RoBERTa*+': {'Prec': 54.2, 'Rec': 59.5, 'F1': 56.8},
        "DeBERTa+": {"Prec": 66.3, "Rec": 59.7, "F1": 62.8},
        "Supervised": {"F1": 71.3},
    },
    "EE": {
        "RoBERTa": {"Prec": 23.8, "Rec": 63.0, "F1": 34.5},
        'RoBERTa*': {'Prec': 23.5, 'Rec': 60.8, 'F1': 33.9},
        "DeBERTa": {"Prec": 12.9, "Rec": 60.3, "F1": 21.2},
        "RoBERTa+": {"Prec": 32.0, "Rec": 52.9, "F1": 39.9},
        'RoBERTa*+': {'Prec': 25.1, 'Rec': 58.6, 'F1': 35.1},
        "DeBERTa+": {"Prec": 13.0, "Rec": 55.8, "F1": 21.1},
        "Supervised": {"F1": 73.4},
    },
    "EAE": {
        "RoBERTa": {"Prec": 20.5, "Rec": 60.9, "F1": 30.7},
        'RoBERTa*': {'Prec': 30.1, 'Rec': 63.2, 'F1': 40.8},
        "DeBERTa": {"Prec": 20.0, "Rec": 31.9, "F1": 24.6},
        "RoBERTa+": {"Prec": 25.8, "Rec": 40.1, "F1": 31.4},
        'RoBERTa*+': {'Prec': 31.1, 'Rec': 58.3, 'F1': 40.6},
        "DeBERTa+": {"Prec": 28.9, "Rec": 17.5, "F1": 21.8},
        "Supervised": {"F1": 72.1},
    },
}

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.8), layout="constrained")

for i, (task, data) in enumerate(results.items()):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.bar(
        i*SHIFT - (BAR_WIDTH*2 + SPACING),
        data["Supervised"]["F1"],
        color=PRIMARY_COLOR,
        label="Supervised (100%)" if not i else None,
        alpha=0.25,
        width=BAR_WIDTH,
        edgecolor=PRIMARY_COLOR,
        hatch="//",
    )

    for j, model in enumerate(["RoBERTa", "DeBERTa"]):
        ax.bar(
            # (i*BLOCK_SPACING + SHIFT) + j*(2*BAR_WIDTH + SPACING) - SHIFT/2,
            i*SHIFT + j*(BAR_WIDTH*2 + SPACING) - BAR_WIDTH,
            data[model]["F1"],
            color=SECONDARY_COLOR if model == "RoBERTa" else TERTIARY_COLOR,
            label=model if not i else None,
            alpha=0.5,
            width=BAR_WIDTH,
            edgecolor=SECONDARY_COLOR if model == "RoBERTa" else TERTIARY_COLOR,
            hatch="//",
        )

        ax.bar(
            i*SHIFT + j*(BAR_WIDTH*2 + SPACING),
            data[model + "+"]["F1"],
            color=SECONDARY_COLOR if model == "RoBERTa" else TERTIARY_COLOR,
            label=model + r" ($\tau$ opt)" if not i else None,
            alpha=0.25,
            width=BAR_WIDTH,
            edgecolor=SECONDARY_COLOR if model == "RoBERTa" else TERTIARY_COLOR,
            hatch="//",
        )
    
    

ax.set_xticks(
    [0, 3, 6, 9],
    labels=["\nNER\nCoNLL 2003", "\nRE\nTACRED", "\nEE\nACE05", "\nEAE\nACE05"],
    fontsize=14,
)
ax.grid(axis="y", linestyle="--", alpha=0.75)
ax.grid(axis="x", alpha=0.0)
ax.set_ylabel("F1 Score", fontsize=14)

fig.legend(loc="outside upper center", fontsize=14, ncol=5)

plt.savefig("images/zero_shot_results.pgf", bbox_inches="tight")

#############################################################3

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3.8), layout="constrained")

# for i, (task, data) in enumerate(results.items()):
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["bottom"].set_visible(False)
#     ax.spines["left"].set_visible(False)

#     for j, model in enumerate(["RoBERTa+", "RoBERTa*+"]):
#         print(2 * i + -.5*j, "||", i, j, model, )
#         ax.bar(
#             2 * i + -.5*(not j),
#             data[model]["F1"],
#             color=PRIMARY_COLOR if model == "RoBERTa+" else SECONDARY_COLOR,
#             label=model.replace("RoBERTa*+", "NLI").replace("RoBERTa+", r"NLI\textsubscript{MNLI-only}") if not i else None,
#             alpha=0.5,
#             width=0.5,
#             edgecolor=PRIMARY_COLOR if model == "RoBERTa+" else SECONDARY_COLOR,
#             hatch="//",
#         )

# ax.set_xticks(
#     np.arange(4)*2 - .25,
#     labels=["\nNER\nCoNLL 2003", "\nRE\nTACRED", "\nEE\nACE05", "\nEAE\nACE05"],
#     fontsize=14,
# )
# ax.grid(axis="y", linestyle="--", alpha=0.75)
# ax.grid(axis="x", alpha=0.0)
# ax.set_ylabel("F1 Score", fontsize=14)

# fig.legend(loc="outside upper center", fontsize=14, ncol=4)

# plt.savefig("images/nli_data_zero_shot_results.png", bbox_inches="tight", dpi=300)
