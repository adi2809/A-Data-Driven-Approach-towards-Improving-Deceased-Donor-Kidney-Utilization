from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, confusion_matrix, roc_auc_score

from .utils import ensure_parent


sns.set_theme(style="whitegrid")


PALETTE = {
    "navy": "#23486A",
    "teal": "#2A9D8F",
    "coral": "#E76F51",
    "mint": "#A8DADC",
    "rose": "#E63946",
    "slate": "#5C677D",
}

LABEL_DISPLAY_MAP = {
    "none": "No-yes run",
    "localizable_observed_y": "Observed yes run",
    "censored_positive": "Hidden placement run",
    "audit_orphan_y": "Yes-with-no-link run",
    "early": "Early",
    "mid": "Middle",
    "late": "Late",
}


def _friendly_label(value: str) -> str:
    return LABEL_DISPLAY_MAP.get(value, str(value).replace("_", " ").title())


def _friendly_feature_name(value: str) -> str:
    return str(value).replace("_", " ").replace(" nm", " NM").replace(" kdpi", " KDPI").title()


def _style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#d9dee7", linewidth=0.8)
    ax.set_axisbelow(True)


def _annotate_vertical_bars(
    ax: plt.Axes,
    bars: Any,
    fmt: str = "{:.3f}",
    offset: float = 0.01,
    rotation: int = 0,
) -> None:
    for bar in bars:
        value = float(bar.get_height())
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=rotation,
            color=PALETTE["navy"],
        )


def _save_figure(output_path: Path) -> None:
    ensure_parent(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="#fbfbf8")
    plt.close()


def plot_data_qa(match_labels: pd.DataFrame, output_path: Path) -> None:
    labels = match_labels.copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    state_counts = (
        labels.groupby(["match_year", "run_state"]).size().unstack(fill_value=0).sort_index()
    )
    state_counts.plot(kind="bar", stacked=True, ax=axes[0, 0], colormap="tab20c")
    axes[0, 0].set_title("Run States by Match Year")
    axes[0, 0].set_xlabel("Match Year")
    axes[0, 0].set_ylabel("Runs")

    for run_state, frame in labels.groupby("run_state"):
        ordered = np.sort(frame["run_len"].astype(float).to_numpy())
        if len(ordered) == 0:
            continue
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        axes[0, 1].plot(ordered, cdf, label=run_state)
    axes[0, 1].set_title("Run Length CDF by Run State")
    axes[0, 1].set_xlabel("Run Length")
    axes[0, 1].set_ylabel("CDF")
    axes[0, 1].legend()

    localizable = labels.loc[labels["run_state"] == "localizable_observed_y"].copy()
    if not localizable.empty:
        axes[1, 0].hist(localizable["first_observed_y_rank"].dropna(), bins=30, color="#457b9d")
        axes[1, 0].set_title("First Observed Y Rank")
        axes[1, 0].set_xlabel("Offer Rank")
        axes[1, 0].set_ylabel("Count")

        axes[1, 1].hist(
            localizable["normalized_first_observed_y_rank"].dropna(),
            bins=30,
            color="#1d3557",
        )
        axes[1, 1].set_title("Normalized First Observed Y Rank")
        axes[1, 1].set_xlabel("Normalized Offer Rank")
        axes[1, 1].set_ylabel("Count")
    else:
        axes[1, 0].text(0.5, 0.5, "No localizable runs", ha="center", va="center")
        axes[1, 1].text(0.5, 0.5, "No localizable runs", ha="center", va="center")

    fig.suptitle("Data QA", y=0.98)
    _save_figure(output_path)


def plot_offerpred_diagnostics(eval_rows: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    y_true = eval_rows["offerpred_target"].astype(int)
    y_score = eval_rows["offerpred_score"].astype(float)
    negative_scores = y_score.loc[y_true == 0]
    positive_scores = y_score.loc[y_true == 1]
    total_rows = int(len(eval_rows))
    negative_count = int(len(negative_scores))
    positive_count = int(len(positive_scores))
    positive_rate = 0.0 if total_rows == 0 else positive_count / total_rows

    if y_true.nunique() > 1:
        roc_auc = roc_auc_score(y_true, y_score)
        RocCurveDisplay.from_predictions(
            y_true,
            y_score,
            ax=axes[0, 0],
            name="Classifier",
            color=PALETTE["navy"],
        )
        axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color=PALETTE["slate"], linewidth=1.2)
        axes[0, 0].set_title("OfferPred ROC")
        axes[0, 0].legend([f"Classifier (AUC = {roc_auc:.3f})"], loc="lower right")

        average_precision = average_precision_score(y_true, y_score)
        PrecisionRecallDisplay.from_predictions(
            y_true,
            y_score,
            ax=axes[0, 1],
            name="Classifier",
            color=PALETTE["teal"],
        )
        axes[0, 1].axhline(
            positive_rate,
            linestyle="--",
            color=PALETTE["slate"],
            linewidth=1.2,
            label=f"Base Y rate = {positive_rate:.3%}",
        )
        axes[0, 1].set_title("OfferPred Precision-Recall")
        axes[0, 1].legend(
            [f"Classifier (AP = {average_precision:.3f})", f"Base Y rate = {positive_rate:.3%}"],
            loc="lower left",
        )

        bucket_frame = pd.DataFrame({"y_true": y_true, "y_score": y_score}).copy()
        bucket_frame["score_decile"] = pd.qcut(
            bucket_frame["y_score"],
            q=10,
            labels=False,
            duplicates="drop",
        )
        bucket_summary = (
            bucket_frame.groupby("score_decile", observed=True)
            .agg(
                row_count=("y_true", "size"),
                y_rate=("y_true", "mean"),
                score_min=("y_score", "min"),
                score_max=("y_score", "max"),
            )
            .reset_index()
        )
        bucket_summary["score_decile"] = bucket_summary["score_decile"].astype(int) + 1

        bars = axes[1, 0].bar(
            bucket_summary["score_decile"],
            bucket_summary["y_rate"],
            color=PALETTE["coral"],
            alpha=0.9,
            edgecolor="white",
            linewidth=1.0,
        )
        axes[1, 0].axhline(
            positive_rate,
            linestyle="--",
            color=PALETTE["slate"],
            linewidth=1.5,
            label=f"Overall Y rate = {positive_rate:.3%}",
        )
        axes[1, 0].set_xticks(bucket_summary["score_decile"])
        axes[1, 0].set_xlabel("Score decile (1 = lowest scores, 10 = highest scores)")
        axes[1, 0].set_ylabel("Actual Y rate")
        axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[1, 0].set_title("OfferPred Y Rate By Score Decile")
        axes[1, 0].legend(loc="upper left")
        y_max = max(float(bucket_summary["y_rate"].max()), positive_rate) * 1.22 if not bucket_summary.empty else 0.01
        axes[1, 0].set_ylim(0, y_max)
        for bar, rate in zip(bars, bucket_summary["y_rate"]):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                float(rate) + y_max * 0.015,
                f"{float(rate):.2%}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color=PALETTE["navy"],
            )
        axes[1, 0].text(
            0.03,
            0.97,
            f"Eval rows: {total_rows:,}\nN rows: {negative_count:,}\nY rows: {positive_count:,} ({positive_rate:.3%})",
            transform=axes[1, 0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PALETTE["mint"]},
        )
        axes[1, 0].text(
            0.97,
            0.97,
            "Each bar holds about 10% of rows.\nMoving right means higher model scores.\nA good model puts higher Y rates on the right.",
            transform=axes[1, 0].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PALETTE["mint"]},
        )
    else:
        for axis in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            axis.text(0.5, 0.5, "Need both classes", ha="center", va="center")
            axis.set_axis_off()

    bins = np.linspace(0.0, 1.0, 31)
    if negative_count > 0:
        axes[1, 1].hist(
            negative_scores,
            bins=bins,
            weights=np.ones(negative_count) / negative_count,
            color=PALETTE["mint"],
            alpha=0.85,
            edgecolor="white",
            label=f"N ({negative_count:,})",
        )
        axes[1, 1].axvline(
            float(negative_scores.median()),
            color=PALETTE["navy"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
    if positive_count > 0:
        axes[1, 1].hist(
            positive_scores,
            bins=bins,
            weights=np.ones(positive_count) / positive_count,
            histtype="step",
            linewidth=2.5,
            color=PALETTE["rose"],
            label=f"Y ({positive_count:,})",
        )
        axes[1, 1].axvline(
            float(positive_scores.median()),
            color=PALETTE["rose"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
    axes[1, 1].set_title("OfferPred Score Distribution By Class")
    axes[1, 1].set_xlabel("Predicted P(observed Y)")
    axes[1, 1].set_ylabel("Share of each class")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].legend()
    axes[1, 1].text(
        0.03,
        0.97,
        "Each class is normalized separately.\nThis makes the Y pattern visible even though Y rows are rare.",
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PALETTE["mint"]},
    )

    fig.suptitle("OfferPred Diagnostics", y=0.98)
    _save_figure(output_path)


def plot_offerpred_topk(eval_rows: pd.DataFrame, output_path: Path, ks: tuple[int, ...] = (1, 3, 5, 10)) -> None:
    rows = eval_rows.copy()
    rows = rows.sort_values(["match_id", "offerpred_score", "offer_rank"], ascending=[True, False, True])

    metrics: list[dict[str, float]] = []
    for k in ks:
        topk = rows.groupby("match_id").head(k)
        positives_per_match = topk.groupby("match_id")["offerpred_target"].max()
        precision_per_match = topk.groupby("match_id")["offerpred_target"].mean()
        metrics.append(
            {
                "k": k,
                "recall_at_k": float(positives_per_match.mean()) if not positives_per_match.empty else 0.0,
                "precision_at_k": float(precision_per_match.mean()) if not precision_per_match.empty else 0.0,
            }
        )

    frame = pd.DataFrame(metrics)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    fig.patch.set_facecolor("#fbfbf8")
    x = np.arange(len(frame))
    width = 0.36
    recall_bars = ax.bar(
        x - width / 2,
        frame["recall_at_k"],
        width=width,
        color=PALETTE["navy"],
        label="Recall@k",
    )
    precision_bars = ax.bar(
        x + width / 2,
        frame["precision_at_k"],
        width=width,
        color=PALETTE["teal"],
        label="Precision@k",
    )
    ax.set_xticks(x, [f"Top {k}" for k in frame["k"]])
    ax.set_xlabel("Rows kept per run")
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("OfferPred Retrieval Quality By Top-k", pad=12, fontsize=15, weight="bold")
    _style_axes(ax)
    _annotate_vertical_bars(ax, recall_bars, fmt="{:.1%}", offset=0.015)
    _annotate_vertical_bars(ax, precision_bars, fmt="{:.1%}", offset=0.015)
    ax.legend(frameon=True, loc="upper left")
    _save_figure(output_path)


def plot_offerpred_yearly_metrics(yearly_metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#fbfbf8")
    frame = yearly_metrics.copy()
    if frame.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No yearly metrics", ha="center", va="center")
            axis.set_axis_off()
        _save_figure(output_path)
        return

    frame["label"] = frame["split"].astype(str).str.title() + "\n" + frame["match_year"].astype(str)
    metric_specs = [
        ("roc_auc", "ROC AUC", PALETTE["navy"], False),
        ("average_precision", "Average Precision", PALETTE["teal"], True),
        ("positive_rate", "Observed Y Rate", PALETTE["mint"], True),
    ]

    for axis, (column, title, color, use_percent) in zip(axes, metric_specs):
        bars = axis.bar(frame["label"], frame[column], color=color, edgecolor="white", linewidth=1.2)
        axis.set_title(title, fontsize=14, weight="bold")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=0)
        _style_axes(axis)
        y_max = float(frame[column].max()) if not frame.empty else 1.0
        axis.set_ylim(0, max(y_max * 1.22, 0.05 if column == "positive_rate" else 0.4))
        if use_percent:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            _annotate_vertical_bars(axis, bars, fmt="{:.1%}", offset=max(y_max * 0.02, 0.002), rotation=90)
        else:
            _annotate_vertical_bars(axis, bars, fmt="{:.3f}", offset=max(y_max * 0.02, 0.01), rotation=90)
    _save_figure(output_path)


def plot_offerpred_feature_importance(feature_importance: pd.DataFrame, output_path: Path, top_n: int = 25) -> None:
    frame = feature_importance.copy().sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10.8, 8.8))
    fig.patch.set_facecolor("#fbfbf8")
    if frame.empty:
        ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(output_path)
        return
    frame = frame.iloc[::-1].copy()
    frame["feature_label"] = frame["feature"].map(_friendly_feature_name)
    bars = ax.barh(
        frame["feature_label"],
        frame["importance"],
        color=PALETTE["coral"],
        edgecolor="white",
        linewidth=1.0,
    )
    for bar, value in zip(bars, frame["importance"]):
        ax.text(
            float(value) + max(float(frame["importance"].max()) * 0.012, 0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color=PALETTE["navy"],
        )
    ax.set_title("OfferPred Most Influential Features", fontsize=15, weight="bold", pad=12)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    _style_axes(ax, grid_axis="x")
    _save_figure(output_path)


def plot_confusion(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
    title: str,
    output_path: Path,
    *,
    subtitle: str | None = None,
    footnote: str | None = None,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    display_labels = [LABEL_DISPLAY_MAP.get(label, str(label).replace("_", " ").title()) for label in labels]
    annotation = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            annotation[i, j] = f"{normalized[i, j]:.1%}\n({matrix[i, j]:,})"

    fig_width = 7.2 if len(labels) <= 2 else 8.2
    fig_height = 6.6 if len(labels) <= 2 else 6.9
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("#fbfbf8")
    heatmap = sns.heatmap(
        normalized,
        annot=annotation,
        fmt="",
        cmap=sns.blend_palette(["#F5F8FC", "#DCEBF2", PALETTE["navy"]], as_cmap=True),
        cbar=False,
        square=True,
        linewidths=1.2,
        linecolor="white",
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        annot_kws={"fontsize": 11, "weight": "bold"},
    )
    for text, value in zip(heatmap.texts, normalized.flatten(order="C")):
        text.set_color("white" if value >= 0.58 else PALETTE["navy"])
    ax.set_title("")
    ax.set_xlabel("Predicted class", labelpad=10)
    ax.set_ylabel("True class", labelpad=10)
    ax.tick_params(axis="x", rotation=0, labelsize=11)
    ax.tick_params(axis="y", rotation=0, labelsize=11)
    fig.subplots_adjust(
        top=0.77 if subtitle else 0.83,
        bottom=0.22 if footnote else 0.16,
        left=0.19 if len(labels) <= 2 else 0.24,
        right=0.97,
    )
    fig.text(
        0.12,
        0.96,
        title,
        ha="left",
        va="top",
        fontsize=16,
        weight="bold",
        color=PALETTE["navy"],
    )
    if subtitle:
        fig.text(
            0.12,
            0.91,
            subtitle,
            ha="left",
            va="top",
            fontsize=10.5,
            color=PALETTE["slate"],
        )
    note = footnote or "Cells show row-normalized percentages with raw counts in parentheses."
    fig.text(
        0.12,
        0.06,
        note,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=PALETTE["slate"],
    )
    ensure_parent(output_path)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="#fbfbf8")
    plt.close(fig)


def plot_discardpred_score_mass(discardpred_eval: pd.DataFrame, output_path: Path) -> None:
    if "offerpred_score_max" not in discardpred_eval.columns:
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        fig.patch.set_facecolor("#fbfbf8")
        ax.axis("off")
        ax.text(
            0.5,
            0.58,
            "OfferPred run-score summaries are not part of this DiscardPred feature set.",
            ha="center",
            va="center",
            fontsize=15,
            weight="bold",
            color=PALETTE["navy"],
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.40,
            "This panel is only available for the offline DiscardPred model.",
            ha="center",
            va="center",
            fontsize=11,
            color=PALETTE["slate"],
            transform=ax.transAxes,
        )
        _save_figure(output_path)
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    fig.patch.set_facecolor("#fbfbf8")
    order = ["placed", "discard"]
    labels = [_friendly_label(x) for x in order]
    plot_frame = discardpred_eval.copy()
    plot_frame["outcome"] = plot_frame["discard_target"].astype(int).map({0: "placed", 1: "discard"})
    sns.violinplot(
        data=plot_frame,
        x="outcome",
        y="offerpred_score_max",
        order=order,
        color=PALETTE["mint"],
        inner=None,
        cut=0,
        linewidth=0.8,
        ax=ax,
    )
    sns.boxplot(
        data=plot_frame,
        x="outcome",
        y="offerpred_score_max",
        order=order,
        width=0.22,
        showcaps=True,
        boxprops={"facecolor": "white", "zorder": 3},
        whiskerprops={"linewidth": 1.2},
        medianprops={"color": PALETTE["rose"], "linewidth": 2},
        ax=ax,
    )
    medians = plot_frame.groupby("outcome")["offerpred_score_max"].median().reindex(order)
    for idx, value in enumerate(medians):
        if pd.notna(value):
            ax.text(idx, float(value) + 0.02, f"median {float(value):.3f}", ha="center", va="bottom", fontsize=9, color=PALETTE["navy"])
    ax.set_title("How Strong The Best OfferPred Row Looks In Each Run Type", fontsize=15, weight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Highest OfferPred row score within the run")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 1.02)
    _style_axes(ax)
    _save_figure(output_path)


def plot_locationpred_localizer(
    localizer_eval: pd.DataFrame,
    timing_eval: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    fig.patch.set_facecolor("#fbfbf8")

    deltas = [0, 1, 5, 10, 25, 50]
    if not localizer_eval.empty:
        hits = []
        errors = (localizer_eval["pred_rank"] - localizer_eval["true_rank"]).abs().astype(float)
        for delta in deltas:
            hits.append(float((errors <= delta).mean()))
        axes[0].plot(deltas, hits, marker="o", color=PALETTE["navy"], linewidth=2.2)
        for delta, hit in zip(deltas, hits):
            axes[0].text(delta, hit + 0.02, f"{hit:.1%}", ha="center", va="bottom", fontsize=9)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Hit Rate By Allowed Distance", fontsize=14, weight="bold")
        axes[0].set_xlabel("Delta")
        axes[0].set_ylabel("Hit rate")
        axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
        _style_axes(axes[0])

        ordered_errors = np.sort(errors.to_numpy())
        cdf = np.arange(1, len(ordered_errors) + 1) / len(ordered_errors)
        axes[1].plot(ordered_errors, cdf, color=PALETTE["teal"], linewidth=2.2)
        axes[1].set_title("Absolute Error Cumulative Curve", fontsize=14, weight="bold")
        axes[1].set_xlabel("|pred rank - true rank|")
        axes[1].set_ylabel("CDF")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
        _style_axes(axes[1])
    else:
        axes[0].text(0.5, 0.5, "No localizer evaluation rows", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No localizer evaluation rows", ha="center", va="center")

    if not timing_eval.empty:
        matrix = confusion_matrix(
            timing_eval["true_timing"],
            timing_eval["pred_timing"],
            labels=["early", "mid", "late"],
        )
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
        annotation = np.empty_like(matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                annotation[i, j] = f"{normalized[i, j]:.1%}\n({matrix[i, j]:,})"
        sns.heatmap(
            normalized,
            annot=annotation,
            fmt="",
            cmap=sns.light_palette(PALETTE["navy"], as_cmap=True),
            cbar=False,
            xticklabels=[_friendly_label(x) for x in ["early", "mid", "late"]],
            yticklabels=[_friendly_label(x) for x in ["early", "mid", "late"]],
            ax=axes[2],
            linewidths=1.2,
            linecolor="white",
            square=True,
        )
        axes[2].set_title("Timing Bucket Confusion", fontsize=14, weight="bold")
        axes[2].set_xlabel("Predicted bucket")
        axes[2].set_ylabel("True bucket")
    else:
        axes[2].text(0.5, 0.5, "No timing evaluation rows", ha="center", va="center")

    fig.suptitle("LocationPred Localizer", y=1.02, fontsize=18, weight="bold")
    _save_figure(output_path)


def plot_validation_sweep(
    validation_metrics: list[dict[str, Any]],
    output_path: Path,
) -> None:
    frame = pd.DataFrame(validation_metrics)
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.patch.set_facecolor("#fbfbf8")
    metrics = [
        ("delta_0_hit_rate", "Exact match", PALETTE["rose"]),
        ("delta_5_hit_rate", "Within 5", PALETTE["navy"]),
        ("delta_10_hit_rate", "Within 10", PALETTE["teal"]),
    ]
    x = np.arange(len(frame))
    width = 0.22
    for offset, (column, label, color) in zip([-width, 0, width], metrics):
        if column not in frame.columns:
            continue
        bars = ax.bar(x + offset, frame[column], width=width, label=label, color=color)
        _annotate_vertical_bars(ax, bars, fmt="{:.1%}", offset=0.01, rotation=90)
    ax.set_xticks(x, frame["model"].astype(str).str.replace("_", " ", regex=False).str.title())
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Hit rate")
    ax.set_xlabel("")
    ax.set_title("Validation Comparison Across Final Models And Baselines", fontsize=15, weight="bold")
    _style_axes(ax)
    ax.legend(frameon=True, loc="upper right")
    _save_figure(output_path)


def plot_pipeline_dashboard(
    summary_rows: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.patch.set_facecolor("#fbfbf8")
    labels = summary_rows["model"].astype(str).str.replace("_", " ", regex=False).str.title()

    bars0 = axes[0].bar(labels, summary_rows["delta_5_hit_rate"], color=PALETTE["navy"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Final Accuracy Within 5 Rows", fontsize=14, weight="bold")
    axes[0].set_ylabel("Hit rate")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    _style_axes(axes[0])
    _annotate_vertical_bars(axes[0], bars0, fmt="{:.1%}", offset=0.015, rotation=90)

    bars1 = axes[1].bar(labels, summary_rows["coverage"], color=PALETTE["mint"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Share Of Runs Sent To Localization", fontsize=14, weight="bold")
    axes[1].set_ylabel("Coverage")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    _style_axes(axes[1])
    _annotate_vertical_bars(axes[1], bars1, fmt="{:.1%}", offset=0.015, rotation=90)

    fig.suptitle("Whole Pipeline Dashboard", y=1.02, fontsize=18, weight="bold")
    _save_figure(output_path)


def plot_report_scorecard(
    metric_rows: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = metric_rows.copy()
    fig, ax = plt.subplots(figsize=(12.5, max(6.5, 0.42 * max(len(frame), 1))))
    fig.patch.set_facecolor("#fbfbf8")
    if frame.empty:
        ax.text(0.5, 0.5, "No summary metrics available", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(output_path)
        return
    model_map = {"offerpred": "OfferPred", "discardpred": "DiscardPred", "locationpred": "LocationPred"}
    metric_map = {
        "roc_auc": "ROC AUC",
        "average_precision": "Average Precision",
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "macro_ovr_roc_auc": "Macro OVR ROC AUC",
        "delta_0_hit_rate": "Exact match",
        "delta_1_hit_rate": "Within 1 row",
        "delta_5_hit_rate": "Within 5 rows",
        "delta_10_hit_rate": "Within 10 rows",
        "delta_25_hit_rate": "Within 25 rows",
        "delta_50_hit_rate": "Within 50 rows",
    }
    split_map = {
        "validation": "Validation",
        "test": "Test",
        "validation_route": "Val route",
        "test_route": "Test route",
        "validation_timing": "Val timing",
        "test_timing": "Test timing",
    }
    frame["metric_label"] = (
        frame["model"].map(model_map).fillna(frame["model"].astype(str))
        + " | "
        + frame["metric"].map(metric_map).fillna(frame["metric"].astype(str))
    )
    frame["split_label"] = frame["split"].map(split_map).fillna(frame["split"].astype(str))
    pivot = (
        frame.pivot_table(index="metric_label", columns="split_label", values="value", aggfunc="first")
        .sort_index(axis=0)
    )
    ordered_columns = [c for c in ["Validation", "Test", "Val route", "Test route", "Val timing", "Test timing"] if c in pivot.columns]
    if ordered_columns:
        pivot = pivot.reindex(columns=ordered_columns)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=sns.color_palette(["#f7f4ea", "#e4c78a", "#c9873a", "#8f4c18"], as_cmap=True),
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Value"},
        linewidths=0.8,
        linecolor="white",
        annot_kws={"fontsize": 9},
    )
    ax.set_title("Consolidated Scorecard", fontsize=16, weight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save_figure(output_path)


def plot_locationpred_error_analysis(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))
    fig.patch.set_facecolor("#fbfbf8")
    deltas = [0, 1, 5, 10, 25, 50]
    split_frames = [("Validation", validation_predictions, PALETTE["navy"]), ("Test", test_predictions, PALETTE["coral"])]

    for split_name, frame, color in split_frames:
        if frame.empty:
            continue
        pred_rank = pd.to_numeric(frame["predicted_rank"], errors="coerce")
        true_rank = pd.to_numeric(frame["first_observed_y_rank"], errors="coerce")
        errors = (pred_rank - true_rank).abs().astype(float).fillna(np.inf)
        hits = [float((errors <= delta).mean()) for delta in deltas]
        axes[0, 0].plot(deltas, hits, marker="o", label=split_name, color=color, linewidth=2.2)
        axes[0, 0].text(deltas[-2], hits[-2] + 0.02, f"{split_name} Δ25 {hits[-2]:.1%}", color=color, fontsize=9)
        ordered_errors = np.sort(errors.to_numpy())
        if len(ordered_errors) > 0:
            cdf = np.arange(1, len(ordered_errors) + 1) / len(ordered_errors)
            axes[0, 1].plot(ordered_errors, cdf, label=split_name, color=color, linewidth=2.2)

    axes[0, 0].set_title("How Close The Final Predicted Row Is", fontsize=14, weight="bold")
    axes[0, 0].set_xlabel("Allowed distance from the true row")
    axes[0, 0].set_ylabel("Hit rate")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].legend(frameon=True)
    _style_axes(axes[0, 0])

    axes[0, 1].set_title("Cumulative Error Distribution", fontsize=14, weight="bold")
    axes[0, 1].set_xlabel("Absolute row error")
    axes[0, 1].set_ylabel("CDF")
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].legend(frameon=True)
    _style_axes(axes[0, 1])

    decision_frame = pd.concat(
        [frame.assign(split=split_name) for split_name, frame, _ in split_frames if not frame.empty],
        ignore_index=True,
    )
    if not decision_frame.empty:
        decision_shares = (
            decision_frame.groupby(["split", "decision"]).size().unstack(fill_value=0).reindex(
                index=[split for split, _, _ in split_frames if split in decision_frame["split"].unique()],
                columns=["discard", "localize"],
                fill_value=0,
            )
        )
        decision_shares = decision_shares.div(decision_shares.sum(axis=1), axis=0)
        bottom = np.zeros(len(decision_shares))
        colors = [PALETTE["coral"], PALETTE["navy"]]
        for column, color in zip(decision_shares.columns, colors):
            values = decision_shares[column].to_numpy()
            bars = axes[1, 0].bar(decision_shares.index, values, bottom=bottom, label=_friendly_label(column), color=color)
            for idx, (bar, value) in enumerate(zip(bars, values)):
                if value >= 0.06:
                    axes[1, 0].text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom[idx] + value / 2,
                        f"{value:.1%}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if color in [PALETTE["navy"], PALETTE["coral"]] else PALETTE["navy"],
                    )
            bottom += values
        axes[1, 0].set_title("What The Final Pipeline Decides To Do", fontsize=14, weight="bold")
        axes[1, 0].set_ylabel("Share of runs")
        axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[1, 0].legend(frameon=True, loc="upper right")
        _style_axes(axes[1, 0])

        bucket_rows = []
        for split_name, frame, color in split_frames:
            if frame.empty:
                continue
            localizable = frame.loc[frame["route_target"] == "localizable_observed_y"].copy()
            if localizable.empty:
                continue
            bucket = pd.cut(
                pd.to_numeric(localizable["run_len"], errors="coerce"),
                bins=[0, 50, 100, 250, 500, np.inf],
                labels=["1-50", "51-100", "101-250", "251-500", "500+"],
            )
            score = localizable.assign(run_bucket=bucket).groupby("run_bucket", observed=False).apply(
                lambda g: float(((pd.to_numeric(g["predicted_rank"], errors="coerce") - pd.to_numeric(g["first_observed_y_rank"], errors="coerce")).abs() <= 5).mean()),
                include_groups=False,
            )
            for bucket_name, value in score.items():
                bucket_rows.append({"split": split_name, "run_bucket": bucket_name, "delta_5": value})

        bucket_frame = pd.DataFrame(bucket_rows)
        if not bucket_frame.empty:
            x = np.arange(bucket_frame["run_bucket"].nunique())
            bucket_order = ["1-50", "51-100", "101-250", "251-500", "500+"]
            width = 0.36
            for offset, (split_name, color) in zip([-width / 2, width / 2], [(split_frames[0][0], split_frames[0][2]), (split_frames[1][0], split_frames[1][2])]):
                sub = bucket_frame[bucket_frame["split"] == split_name].set_index("run_bucket").reindex(bucket_order)
                bars = axes[1, 1].bar(x + offset, sub["delta_5"], width=width, color=color, label=split_name)
                _annotate_vertical_bars(axes[1, 1], bars, fmt="{:.1%}", offset=0.012, rotation=90)
            axes[1, 1].set_xticks(x, bucket_order)
            axes[1, 1].set_ylim(0, 1.05)
            axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
            axes[1, 1].set_title("Within-5 Accuracy By Run Length", fontsize=14, weight="bold")
            axes[1, 1].set_xlabel("Run length bucket")
            axes[1, 1].set_ylabel("Hit rate")
            axes[1, 1].legend(frameon=True, loc="upper right")
            _style_axes(axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, "No localized rows available", ha="center", va="center")
            axes[1, 1].set_axis_off()
    else:
        for axis in axes.flat:
            axis.text(0.5, 0.5, "No LocationPred predictions available", ha="center", va="center")
            axis.set_axis_off()

    fig.suptitle("LocationPred Error Analysis", y=1.02, fontsize=18, weight="bold")
    _save_figure(output_path)
