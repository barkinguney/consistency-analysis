import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def pretty_label(name: str) -> str:
    return name.replace("_", " ").title()


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.9,
        "grid.color": "#D9D9D9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    }
)

DATASET_PATHS = [
    "data/same_cond_comparison/shen_2009_new.csv",
    "data/same_cond_comparison/davidson_2005_new.csv",
]
DATASET_PATHS = [
    "data/output_unc/hydrogen_output/g00000012.csv",
    "data/output_unc/hydrogen_output/x10000006.csv",
    "data/output_unc/hydrogen_output/x10000039.csv",
]
CUSTOM_LABELS = ["5.9% H$_2$ + 2.9% O$_2$ + 91.2% Ar, HerzlerNaumann2009",
                 "20.0% H$_2$ + 10.0% O$_2$ + 70.0% Ar, FujimotoSuzuki1967",
                 "1.33% H$_2$ + 0.67% O$_2$ + 98.0% Ar, Keromnes2009"] # Example: ["Dataset 1", "Dataset 2", "Dataset 3"]
PLOT_TITLE = " φ = 1.0, P$_5$ ~ 1.4 atm"
USE_LOG_Y = True
OUTPUT_PATH = None  # Example: "plots/custom_ignition_delay.png"

# folder_path = Path("data/output_unc/syngas_output")
# DATASET_PATHS = [
#     folder_path / "x10001041.csv",
#     folder_path / "x10001066.csv",
#     folder_path / "x10001056.csv",
#     folder_path / "x10001024_x.csv",
# ]
# CUSTOM_LABELS = ["1.7% H$_2$ + 15.6% CO + 17.5% O$_2$ + 65.2% N$_2$, Kalitan2007",
#                  "0.4% H$_2$ + 0.6% CO + 1.0% O$_2$ + 98.0% Ar, Krejci2012",
#                  "4.4% H$_2$ + 4.4% CO + 8.7% O$_2$ + 82.5% Ar, HerzlerNaumann2009",
#                  "1.7% H$_2$ + 1.7% CO + 3.5% O$_2$ + 93.1% Ar, Naumann2011"] # Example: ["Dataset 1", "Dataset 2", "Dataset 3"]
# PLOT_TITLE = " φ = 0.5, P$_5$ ~ 15 atm"
# USE_LOG_Y = True
# OUTPUT_PATH = None  # Example: "plots/custom_ignition_delay.png"

# folder_path = Path("data/output_unc/syngas_output")
# DATASET_PATHS = [
#     "data/davidson_2005.csv",
#     "data/shen_2009.csv",
# ]
# CUSTOM_LABELS = ["Davidson2005",
#                  "Shen2009",
#                 ]
# PLOT_TITLE = " Toluene/air, φ = 1.0, P ~ 50 atm"
# USE_LOG_Y = True
# OUTPUT_PATH = None  # Example: "plots/custom_ignition_delay.png"


def plot_datasets(dataset_paths: list[str], output_path: Path, use_log_y: bool, custom_labels: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    if custom_labels and len(custom_labels) != len(dataset_paths):
        raise ValueError(f"CUSTOM_LABELS length ({len(custom_labels)}) must match DATASET_PATHS length ({len(dataset_paths)})")

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

    required_columns = {"T5_K", "ignition_delay_time", "final_expanded"}

    for idx, path_str in enumerate(dataset_paths):
        file_path = Path(path_str)
        data = pd.read_csv(file_path)

        missing = required_columns - set(data.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in {file_path}: {missing_text}")

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        filled_marker = idx % 2 == 0
        marker_face = color if filled_marker else "white"
        
        # Use custom label if provided, otherwise auto-generate from filename
        label = custom_labels[idx] if custom_labels else pretty_label(file_path.stem)

        ax.errorbar(
            1000 / data["T5_K"],
            data["ignition_delay_time"],
            yerr=data["final_expanded"],
            fmt=marker,
            markersize=6,
            color=color,
            mfc=marker_face,
            mec=color,
            mew=1.0,
            ecolor=color,
            elinewidth=1.1,
            capsize=3,
            capthick=1.1,
            alpha=0.95,
            label=label,
        )

    ax.set_xlabel("1000/T, [1000/K]")
    ax.set_ylabel("Ignition Delay, [μs]")
    ax.set_title(PLOT_TITLE, pad=10)

    if use_log_y:
        ax.set_yscale("log")

    legend = ax.legend(loc="best", frameon=True)
    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_edgecolor("#DDDDDD")

    ax.grid(True, which="major", axis="both")
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.6)
    ax.margins(x=0.03, y=0.08)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.show()


def main() -> None:
    if not DATASET_PATHS:
        raise ValueError("DATASET_PATHS is empty. Add at least one CSV file path.")

    if OUTPUT_PATH:
        output_path = Path(OUTPUT_PATH)
    else:
        dataset_names = [Path(path).stem for path in DATASET_PATHS]
        joined = "_".join(dataset_names)
        output_path = Path("plots") / f"{joined}_ignition_delay.png"

    plot_datasets(DATASET_PATHS, output_path, USE_LOG_Y, CUSTOM_LABELS)


if __name__ == "__main__":
    main()
