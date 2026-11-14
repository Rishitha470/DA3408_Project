"""Exploratory data analysis of the raw CDC data (data/raw/diabetes.csv, written by the ingest stage).

Writes to reports/eda/:
  01_class_balance.png            how many people have diabetes / prediabetes
  02_bmi_by_diabetes.png          BMI distribution for people with and without diabetes
  03_rate_by_age.png              share with diabetes in each age group
  04_rate_by_general_health.png   share with diabetes by self-rated general health
  05_correlation_with_target.png  correlation of every feature with the target
  eda_summary.json                the numbers behind the charts

Run:  python scripts/eda.py        (or: mlflow run . -e eda --env-manager local --experiment-name diabetes-risk)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                     # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.schema import FEATURE_NAMES, TARGET  # noqa: E402

RAW = ROOT / "data" / "raw" / "diabetes.csv"
OUT = ROOT / "reports" / "eda"
AGE_LABELS = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64",
              "65-69", "70-74", "75-79", "80+"]
HEALTH_LABELS = ["Excellent", "Very good", "Good", "Fair", "Poor"]
NO, YES = "#4c78a8", "#e45756"                            # colours: no diabetes / diabetes


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)


def class_balance(df: pd.DataFrame) -> dict:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(["No diabetes", "Diabetes or prediabetes"], counts.values, color=[NO, YES])
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n, f"{n:,}\n({n / len(df):.1%})", ha="center", va="bottom")
    ax.set_ylabel("People")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_title("Class balance")
    save(fig, "01_class_balance.png")
    return {"negative": int(counts.get(0, 0)), "positive": int(counts.get(1, 0)),
            "positive_rate": round(float(df[TARGET].mean()), 4)}


def bmi_by_target(df: pd.DataFrame) -> dict:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = range(12, 61)
    for value, colour, label in ((0, NO, "No diabetes"), (1, YES, "Diabetes or prediabetes")):
        ax.hist(df.loc[df[TARGET] == value, "BMI"].clip(upper=60), bins=bins, density=True,
                alpha=0.55, color=colour, label=label)
    ax.set_xlabel("BMI (values above 60 shown at 60)")
    ax.set_ylabel("Share of group")
    ax.set_title("BMI by diabetes status")
    ax.legend()
    save(fig, "02_bmi_by_diabetes.png")
    stats = df.groupby(TARGET)["BMI"].describe()[["mean", "50%", "std"]].round(2)
    return {("diabetes" if k == 1 else "no_diabetes"): {"mean": v["mean"], "median": v["50%"], "std": v["std"]}
            for k, v in stats.iterrows()}


def rate_by(df: pd.DataFrame, column: str, labels: list, title: str, xlabel: str, name: str) -> dict:
    rates = df.groupby(column)[TARGET].mean()
    fig, ax = plt.subplots(figsize=(7 if len(labels) > 6 else 5, 3.5))
    ax.bar([labels[int(i) - 1] for i in rates.index], rates.values * 100, color=YES)
    ax.axhline(df[TARGET].mean() * 100, color="grey", linestyle="--", label="Overall")
    ax.set_ylabel("% with diabetes or prediabetes")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45 if len(labels) > 6 else 0, ha="right" if len(labels) > 6 else "center")
    save(fig, name)
    return {labels[int(i) - 1]: round(float(r), 4) for i, r in rates.items()}


def correlation_with_target(df: pd.DataFrame) -> dict:
    corr = df[FEATURE_NAMES].corrwith(df[TARGET]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(corr.index, corr.values, color=[YES if v > 0 else NO for v in corr.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with diabetes\n(positive = more common with diabetes)")
    ax.set_title("Which answers go together with diabetes")
    save(fig, "05_correlation_with_target.png")
    return {k: round(float(v), 4) for k, v in corr.sort_values(key=abs, ascending=False).items()}


def main() -> int:
    if not RAW.exists():
        print(f"{RAW} not found: run `dvc repro` (or `dvc checkout`) first", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)
    summary = {
        "rows": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_balance": class_balance(df),
        "bmi_by_target": bmi_by_target(df),
        "rate_by_age": rate_by(df, "Age", AGE_LABELS, "Diabetes by age group", "Age group", "03_rate_by_age.png"),
        "rate_by_general_health": rate_by(df, "GenHlth", HEALTH_LABELS, "Diabetes by self-rated general health",
                                          "General health", "04_rate_by_general_health.png"),
        "rate_with_high_bp": round(float(df.loc[df["HighBP"] == 1, TARGET].mean()), 4),
        "rate_without_high_bp": round(float(df.loc[df["HighBP"] == 0, TARGET].mean()), 4),
        "correlation_with_target": correlation_with_target(df),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    top = list(summary["correlation_with_target"].items())[:5]
    print(f"rows {summary['rows']:,} | missing {summary['missing_values']} | duplicates {summary['duplicate_rows']:,}"
          f" | positive rate {summary['class_balance']['positive_rate']:.2%}")
    print("strongest correlations with the target:", ", ".join(f"{k} {v:+.3f}" for k, v in top))
    print(f"wrote 5 charts and eda_summary.json to {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
