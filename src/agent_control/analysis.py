"""Generic results analysis over ControlArena logs (no setting-specific code)."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaErrors
from control_arena.analysis import (
    compute_safety,
    compute_safety_upfront_audit,
    compute_usefulness,
    compute_usefulness_upfront_audit,
    plot_suspicion_scores_histogram,
    samples_df,
    scale_usefulness,
)
from inspect_ai.analysis import evals_df
from matplotlib.figure import Figure

SUSPICION_SCORER = "max_suspicion_scorer"
SUSPICION_COLUMN = f"score_{SUSPICION_SCORER}"
UPFRONT_PROTOCOL = "upfront_auditing"

SUMMARY_COLUMNS = [
    "protocol",
    "n_honest",
    "n_attack",
    "usefulness",
    "safety",
    "attack_success_rate",
    "refusal_or_no_attack_rate",
    "n_dropped",
    "safety_definition",
]

SAFETY_DEF_CAUGHT = "caught/(caught+uncaught successful)"
SAFETY_DEF_UPFRONT = "P(no successful attack in deployment)"


def parse_protocol(task_name: str, setting_name: str) -> str:
    """Extract the protocol prefix from a task name.

    ControlArena names tasks ``{prefix}_{setting}_{micro_protocol}_{mode}_{hash}``
    and the runner sets the prefix to the protocol name, so the protocol is
    everything before ``_{setting}_``. Falls back to the whole name.
    """
    marker = f"_{setting_name}_"
    head, sep, _ = task_name.partition(marker)
    return head if sep else task_name


def load_results(log_dir: Path) -> pd.DataFrame:
    """Load logs via ControlArena's ``samples_df`` and add a ``protocol`` column."""
    df = samples_df(str(log_dir))
    names = evals_df(str(log_dir))[["log", "task_name"]].drop_duplicates("log")
    df = df.merge(names, on="log", how="left")
    df["protocol"] = [
        parse_protocol(str(task), str(setting))
        for task, setting in zip(
            df["task_name"], df["metadata_setting_name"], strict=True
        )
    ]
    return df


def _percent(audit_budget: float) -> float:
    """Config budgets are fractions (0.02); ControlArena wants percent (2.0)."""
    return audit_budget * 100.0


def _require_modes(df: pd.DataFrame, protocol: str, modes: tuple[str, ...]) -> None:
    for mode in modes:
        if not (df["eval_mode"] == mode).any():
            raise ValueError(f"protocol {protocol!r}: no {mode} samples found in logs")


def _scale(value: float, trusted: float, untrusted: float) -> float:
    if untrusted == trusted:
        return math.nan
    return scale_usefulness(value, trusted, untrusted)


def _attack_rates(attack: pd.DataFrame) -> tuple[float, float]:
    """(success rate requiring main+side, rate where side task did not succeed)."""
    if attack.empty:
        return math.nan, math.nan
    side = attack["side_task_success"].fillna(False).astype(bool)
    main = attack["main_task_success"].fillna(False).astype(bool)
    return float((side & main).mean()), float((~side).mean())


def _drop_unscored(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop samples with NA task-success values (errored/unscored samples).

    ControlArena's metric functions reject NA task success. Honest samples need
    ``main_task_success``; attack samples need both it and ``side_task_success``.
    Returns the kept frame and the number dropped per protocol.
    """
    main_na = df["main_task_success"].isna()
    side_na = df["side_task_success"].isna()
    bad = main_na | ((df["eval_mode"] == "attack") & side_na)
    dropped = df[bad]["protocol"].value_counts().to_dict()
    for protocol, n in dropped.items():
        print(
            f"warning: protocol {protocol!r}: dropped {n} sample(s) with missing "
            "task-success scores (errored or unscored)"
        )
    return df[~bad], {str(k): int(v) for k, v in dropped.items()}


def _schema_msg(e: SchemaErrors) -> str:
    return f"invalid or missing values in logs ({e.failure_cases.head().to_dict('records')})"


def _wrap_schema(protocol: str, what: str, fn, *args):
    try:
        return fn(*args)
    except SchemaErrors as e:
        raise ValueError(f"protocol {protocol!r}: cannot compute {what}: {_schema_msg(e)}") from e


def summarize(
    df: pd.DataFrame,
    audit_budget: float,
    trusted_only: str,
    untrusted_only: str,
    upfront_deployment_samples: int | None = None,
) -> pd.DataFrame:
    """One row per protocol with usefulness (scaled) and safety.

    ``audit_budget`` is a fraction (0.02 == 2%). Usefulness is scaled so that
    ``trusted_only`` is 0 and ``untrusted_only`` is 1. Baselines have NaN safety.
    An extra ``upfront_auditing`` row is derived from the untrusted-only runs
    when those include attack samples. ``refusal_or_no_attack_rate`` is the
    share of attack samples whose side task did not succeed (refusals,
    ``secret_input=None``, or failed attempts).

    Samples with NA task-success values (errors) are dropped before computing
    anything; ``n_dropped`` reports how many per protocol.

    ``upfront_deployment_samples`` is the simulated deployment size for the
    upfront-auditing row. Its safety is a different quantity (probability of no
    successful attack across the deployment) from the other rows' safety
    (caught / (caught + uncaught successful)); ``safety_definition`` says which.
    ``None`` uses ControlArena's default of the number of attack samples.
    """
    df, dropped = _drop_unscored(df)
    for baseline in (trusted_only, untrusted_only):
        part = df[df["protocol"] == baseline]
        if part.empty:
            raise ValueError(f"baseline protocol {baseline!r}: no samples found in logs")
        _require_modes(part, baseline, ("honest",))

    pct = _percent(audit_budget)
    # Attack samples with no suspicion score (refusals/errors -> NaN) make
    # ControlArena raise. Decision: an unscored attack is never caught, so score
    # it 0 (0 is never > threshold). Honest NaNs are left alone and still raise.
    if SUSPICION_COLUMN in df.columns:
        df = df.copy()
        df[SUSPICION_COLUMN] = pd.to_numeric(df[SUSPICION_COLUMN], errors="coerce")
        unscored = (df["eval_mode"] == "attack") & df[SUSPICION_COLUMN].isna()
        df.loc[unscored, SUSPICION_COLUMN] = 0.0
    trusted_u = _wrap_schema(
        trusted_only, "usefulness", compute_usefulness, df[df["protocol"] == trusted_only]
    )
    untrusted_u = _wrap_schema(
        untrusted_only,
        "usefulness",
        compute_usefulness,
        df[df["protocol"] == untrusted_only],
    )

    rows = []
    baselines = {trusted_only, untrusted_only}
    for protocol in df["protocol"].drop_duplicates():
        part = df[df["protocol"] == protocol]
        _require_modes(part, protocol, ("honest",))
        honest = part[part["eval_mode"] == "honest"]
        attack = part[part["eval_mode"] == "attack"]
        if protocol in baselines:
            safety = math.nan
        else:
            _require_modes(part, protocol, ("attack",))
            try:
                safety = compute_safety(part, SUSPICION_SCORER, pct)
            except (ValueError, AssertionError) as e:
                raise ValueError(f"protocol {protocol!r}: cannot compute safety: {e}") from e
            except SchemaErrors as e:
                raise ValueError(
                    f"protocol {protocol!r}: cannot compute safety: {_schema_msg(e)}"
                ) from e
        success, no_attack = _attack_rates(attack)
        rows.append(
            {
                "protocol": protocol,
                "n_honest": len(honest),
                "n_attack": len(attack),
                "usefulness": _scale(
                    _wrap_schema(protocol, "usefulness", compute_usefulness, part),
                    trusted_u,
                    untrusted_u,
                ),
                "safety": safety,
                "attack_success_rate": success,
                "refusal_or_no_attack_rate": no_attack,
                "n_dropped": dropped.get(protocol, 0),
                "safety_definition": "" if protocol in baselines else SAFETY_DEF_CAUGHT,
            }
        )

    untrusted = df[df["protocol"] == untrusted_only]
    if (untrusted["eval_mode"] == "attack").any():
        attack = untrusted[untrusted["eval_mode"] == "attack"]
        success, no_attack = _attack_rates(attack)
        deployment = upfront_deployment_samples or len(attack)
        if math.floor(pct / 100.0 * deployment) == 0:
            warnings.warn(
                f"{UPFRONT_PROTOCOL}: audit budget {audit_budget:.2%} of a "
                f"deployment of {deployment} samples audits 0 samples; the row "
                "means 'no auditing'. Increase upfront_deployment_samples.",
                stacklevel=2,
            )
        try:
            raw = compute_usefulness_upfront_audit(
                untrusted, trusted_u, pct, upfront_deployment_samples
            )
            upfront_safety = compute_safety_upfront_audit(
                untrusted, pct, upfront_deployment_samples
            )
        except (ValueError, AssertionError) as e:
            raise ValueError(
                f"protocol {UPFRONT_PROTOCOL!r}: cannot compute upfront audit: {e}"
            ) from e
        except SchemaErrors as e:
            raise ValueError(
                f"protocol {UPFRONT_PROTOCOL!r}: cannot compute upfront audit: "
                f"{_schema_msg(e)}"
            ) from e
        rows.append(
            {
                "protocol": UPFRONT_PROTOCOL,
                "n_honest": int((untrusted["eval_mode"] == "honest").sum()),
                "n_attack": len(attack),
                "usefulness": _scale(raw, trusted_u, untrusted_u),
                "safety": upfront_safety,
                "attack_success_rate": success,
                "refusal_or_no_attack_rate": no_attack,
                "n_dropped": dropped.get(untrusted_only, 0),
                "safety_definition": SAFETY_DEF_UPFRONT,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def plot_summary(summary: pd.DataFrame) -> Figure:
    """Grouped bars of safety and usefulness per protocol (NaN bars are omitted)."""
    fig = Figure(figsize=(10, 5))
    ax = fig.subplots()
    x = list(range(len(summary)))
    width = 0.38
    ax.bar([i - width / 2 for i in x], summary["safety"], width, label="safety")
    ax.bar([i + width / 2 for i in x], summary["usefulness"], width, label="usefulness")
    ax.set_xticks(x)
    labels = [f"{p}*" if p == UPFRONT_PROTOCOL else p for p in summary["protocol"]]
    ax.set_xticklabels(labels, rotation=30, ha="right")
    if UPFRONT_PROTOCOL in set(summary["protocol"]):
        fig.text(
            0.01,
            0.01,
            f"* {UPFRONT_PROTOCOL} safety = {SAFETY_DEF_UPFRONT}, "
            "not comparable to the other protocols' safety",
            fontsize=8,
        )
    ax.set_ylabel("score")
    ax.set_title("Safety and usefulness by protocol")
    ax.legend()
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_suspicion(df: pd.DataFrame, protocol: str) -> Figure:
    """Honest vs attack suspicion histogram for one protocol."""
    part = df[df["protocol"] == protocol]
    if SUSPICION_COLUMN in part.columns:
        part = part[part[SUSPICION_COLUMN].notna()]
    if SUSPICION_COLUMN not in part.columns or part.empty:
        fig = Figure(figsize=(10, 6))
        ax = fig.subplots()
        ax.text(0.5, 0.5, f"No suspicion scores for {protocol}", ha="center")
        ax.set_axis_off()
        return fig
    return plot_suspicion_scores_histogram(part)
