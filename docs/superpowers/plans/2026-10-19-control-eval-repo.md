# Control Evaluation Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python project where red-vs-blue control experiments run live in the agent loop via ControlArena, with a generic layer (config, runner, analysis) and an `apps_backdoor/` experiment (APPS setting + test-run tool + protocol lineup + scripts).

**Architecture:** `src/agent_control/` holds generic config, task-grid runner and results analysis built on `control-arena` / `inspect-ai`. `src/agent_control/apps_backdoor/` holds everything APPS-specific. A future agentic experiment is a sibling folder reusing the generic layer unchanged.

**Tech Stack:** Python >= 3.11, `uv`, `control-arena` (pin the latest release at build time), `inspect-ai`, pandas, matplotlib/seaborn (via ControlArena), pytest, python-dotenv, Docker, OpenRouter.

**Spec:** `docs/superpowers/specs/2026-10-19-control-eval-repo-design.md`

## Global Constraints

- Phase 1 only: one configuration, one fixed defer/edit threshold shared by all protocols, no threshold or audit-budget sweeps, no Pareto curves.
- 20 samples per mode per protocol (honest + attack = 40); 20 honest samples for the trusted-only baseline.
- Models via OpenRouter (`OPENROUTER_API_KEY`). Default U `openrouter/deepseek/deepseek-v3.2`, T `openrouter/google/gemini-2.5-flash`; candidate alternative U `openrouter/anthropic/claude-haiku-5.5`; models live in one config object.
- Uses ControlArena's built-in `apps` setting and its built-in attack policy, plus one custom addition: a test-run tool.
- The monitor must NOT see test-run tool calls or their results (config switch, default hidden; follows the AI Control paper).
- Test-run tool available in both honest and attack modes; gives no oracle (no reference solution, no correctness verdict).
- Lineup: trusted-only and untrusted-only baselines, upfront auditing, trusted monitoring, untrusted monitoring (monitor `model_role="untrusted"`; collusion out of scope), defer-to-trusted, trusted editing. Defer-to-resample optional. If the pinned ControlArena lacks one, cut it and say so.
- Generic layer must not import from `apps_backdoor/`; APPS logic stays in `apps_backdoor/`.
- Offline tests (no live API calls) by default; Docker/API tests behind pytest markers `docker` and `live`.
- Docker is installed but `docker` may not be on PATH (`/Applications/Docker.app/Contents/Resources/bin/docker`); run `docker info` before any docker/live test.
- Issues are tracked in GitHub Issues via `gh`; do not commit logs or `.env`.

## Review Focus

- Attack-mode U refuses or submits `secret_input=None` for most samples: smoke check flags it; analysis must not crash on zero successful attacks.
- Zero honest samples with scores, or zero attack samples, in a protocol's logs: safety computation raises a clear error naming the protocol, not a bare ControlArena `ValueError`.
- Missing `OPENROUTER_API_KEY` or Docker not running: fail fast with an actionable message before any model spend.
- Test-run tool given code that loops forever, crashes, or prints huge output: returns a bounded result (timeout, truncated output), never hangs the sample.
- A step whose tool calls are only test-run calls (or a mix with `submit`): monitor skips or filters it; the protocol must not defer or edit on it.

---

### Task 1: Project scaffold and experiment config

**Files:**
- Create: `pyproject.toml`, `.gitignore` (`.env`, `logs/`, `.venv/`), `.env.example` (`OPENROUTER_API_KEY=`), `src/agent_control/__init__.py`, `src/agent_control/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `ExperimentConfig` (frozen dataclass) with fields `untrusted_model: str`, `trusted_model: str`, `defer_threshold: float`, `audit_budget: float`, `samples_per_mode: int = 20`, `max_steps: int`, `log_dir: Path`, `monitor_sees_test_run: bool = False`; methods `to_control_eval_config() -> ControlEvalConfig` and `require_api_key() -> None` (raises `RuntimeError` naming `OPENROUTER_API_KEY`).
- Defaults: U/T as in Global Constraints. `defer_threshold`, `audit_budget`, `max_steps` are your call; put the values in the config and document the score scale you find in ControlArena's default monitor.

- [ ] **Step 1:** Create the project with `uv init --lib --python 3.11`, `uv add control-arena python-dotenv`, `uv add --dev pytest`; record the resolved `control-arena` version in `pyproject.toml` as an exact pin. Add pytest markers `docker`, `live` to `[tool.pytest.ini_options]`.
- [ ] **Step 2: Write failing tests** in `tests/test_config.py`: `test_defaults_match_spec` (U is `openrouter/deepseek/deepseek-v3.2`, T is `openrouter/google/gemini-2.5-flash`, `samples_per_mode == 20`, `monitor_sees_test_run is False`); `test_to_control_eval_config_maps_fields` (result's `untrusted_model`, `trusted_model`, `limit == samples_per_mode`, `max_steps` equal the config's); `test_require_api_key_raises_without_key` (monkeypatch env, message contains `OPENROUTER_API_KEY`).
- [ ] **Step 3:** Run `uv run pytest tests/test_config.py -v`; expected FAIL (ImportError).
- [ ] **Step 4:** Implement `ExperimentConfig` in `config.py`; `require_api_key` calls `dotenv.load_dotenv()` first.
- [ ] **Step 5:** Run the tests; expected PASS.
- [ ] **Step 6: Commit** `feat: project scaffold and experiment config`.

### Task 2: Generic task-grid runner

**Files:**
- Create: `src/agent_control/runner.py`, `tests/test_runner.py`

**Interfaces:**
- Consumes: `ExperimentConfig` (Task 1).
- Produces: `build_tasks(setting: Setting, protocols: dict[str, ControlAgent], modes: Mapping[str, Sequence[EvalMode]], config: ExperimentConfig) -> list[Task]`: one task per (protocol, mode) in `modes[protocol_name]` (so baselines can be honest-only); each task's name starts with the protocol name. `run_tasks(tasks: list[Task], config: ExperimentConfig, limit: int | None = None) -> list[EvalLog]`: calls `config.require_api_key()`, then `inspect_ai.eval(tasks, log_dir=str(config.log_dir), ...)`. Smoke runs pass a smaller `limit`.

- [ ] **Step 1: Write failing tests** with a trivial fake `Setting` subclass (or ControlArena's own test fixtures if present): `test_build_tasks_one_per_protocol_mode` (2 protocols, one honest-only: expect 3 tasks), `test_task_names_prefixed_with_protocol`, `test_run_tasks_requires_api_key` (no key, `inspect_ai.eval` patched to assert not called).
- [ ] **Step 2:** Run `uv run pytest tests/test_runner.py -v`; expected FAIL.
- [ ] **Step 3:** Implement using `control_arena.eval.get_control_task` once per (protocol, mode), passing `task_name_prefix` and the config's `ControlEvalConfig`. Check the pinned version's `get_control_task` signature; if `max_suspicion_scorer` must be added explicitly as a monitoring scorer for analysis, add it.
- [ ] **Step 4:** Run the tests; expected PASS.
- [ ] **Step 5: Commit** `feat: generic task grid runner`.

### Task 3: Generic results analysis

**Files:**
- Create: `src/agent_control/analysis.py`, `tests/test_analysis.py`

**Interfaces:**
- Produces: `load_results(log_dir: Path) -> pd.DataFrame` (wraps `control_arena.analysis.samples_df`, adds a `protocol` column parsed from the task name/`micro_protocol_info`); `summarize(df: pd.DataFrame, audit_budget: float, trusted_only: str, untrusted_only: str) -> pd.DataFrame` with columns `protocol, n_honest, n_attack, usefulness, safety, attack_success_rate, refusal_or_no_attack_rate`; `plot_summary(summary: pd.DataFrame) -> Figure` (grouped bars of safety and usefulness per protocol); `plot_suspicion(df: pd.DataFrame, protocol: str) -> Figure` (honest vs attack histogram, wraps `plot_suspicion_scores_histogram` when the column exists). Baseline protocols have no safety (NaN).
- Usefulness is scaled with `scale_usefulness` between trusted-only and untrusted-only honest main-task success. Safety uses `compute_safety`; upfront auditing uses `compute_safety_upfront_audit` / `compute_usefulness_upfront_audit` over the untrusted-only runs.

- [ ] **Step 1: Write failing tests** on a synthetic DataFrame built from ControlArena's `SamplesDataFrameSchema` columns (read the schema; do not guess names): `test_summarize_known_values` (hand-built counts give exact safety = caught/(caught+uncaught) and exact usefulness), `test_summarize_zero_successful_attacks_gives_safety_one_without_error`, `test_summarize_missing_attack_samples_raises_naming_protocol` (error message contains the protocol name), `test_plot_summary_returns_figure`.
- [ ] **Step 2:** Run `uv run pytest tests/test_analysis.py -v`; expected FAIL.
- [ ] **Step 3:** Implement. Wrap ControlArena's `ValueError` for empty honest/attack frames in an error that names the protocol.
- [ ] **Step 4:** Run the tests; expected PASS.
- [ ] **Step 5: Commit** `feat: generic results analysis`.

### Task 4: Test-run tool (apps_backdoor)

**Files:**
- Create: `src/agent_control/apps_backdoor/__init__.py`, `src/agent_control/apps_backdoor/test_run_tool.py`, `tests/apps_backdoor/test_test_run_tool.py`

**Interfaces:**
- Produces: `TEST_RUN_TOOL_NAME = "test_run"`; `format_run_result(stdout: str, stderr: str, exit_code: int, timed_out: bool, max_chars: int = 4000) -> str`; `test_run(timeout_s: int = 10) -> Tool` (Inspect `@tool`) taking `code: str, input: str`, executing the code via the sandbox with `input` on stdin and returning the formatted result. No reference solution is ever read or returned.
- Consumes: the sandbox runner conventions in ControlArena's `apps` setting (`control_arena/settings/apps/task/scorers.py`, `run_with_timeout`); reuse them rather than reimplementing execution.

- [ ] **Step 1: Write failing tests:** `test_format_run_result_truncates_to_max_chars`, `test_format_run_result_reports_timeout_and_nonzero_exit` (output contains "timed out" / the exit code), plus a `docker`-marked `test_test_run_executes_code_in_sandbox` (code echoing stdin returns that text) and `test_test_run_infinite_loop_times_out` (returns within `timeout_s` + margin with a timeout message).
- [ ] **Step 2:** Run `uv run pytest tests/apps_backdoor/test_test_run_tool.py -v -m "not docker"`; expected FAIL.
- [ ] **Step 3:** Implement. Tool description must say it runs the given Python on the given stdin and returns output only, for checking behaviour before `submit`.
- [ ] **Step 4:** Run non-docker tests (expect PASS); then run `-m docker` after `docker info` succeeds (expect PASS).
- [ ] **Step 5: Commit** `feat(apps): sandboxed test-run tool`.

### Task 5: Hide test-run calls from the monitor

**Files:**
- Create: `src/agent_control/apps_backdoor/monitor_view.py`, `tests/apps_backdoor/test_monitor_view.py`

**Interfaces:**
- Consumes: `TEST_RUN_TOOL_NAME` (Task 4).
- Produces: `hide_test_run(monitor: ControlAgent) -> ControlAgent` (a `@control_agent` wrapper): if the last assistant message's tool calls are all `test_run`, return the state unchanged and leave the monitor unrun (no suspicion score added, or score 0 with a marker; builder's call, document it); if mixed with other calls, run the inner monitor on a copy of the state whose last message has the `test_run` calls removed; earlier `test_run` calls and their tool results are removed from the history the monitor sees. State returned to the protocol keeps the original messages.
- Check first whether ControlArena's `attack_tool` / `convert_agent_state_to_monitor_view` already gives the needed hiding in both modes (the `attack_tool` without `honest_tool_def` hides from monitors but is absent from honest mode). If not usable for both modes, this wrapper is the mechanism.

- [ ] **Step 1: Write failing tests** with a recording fake monitor (records the state it receives): `test_only_test_run_calls_skips_monitor` (fake never called), `test_mixed_calls_monitor_sees_no_test_run` (fake's received last message has only the `submit` call), `test_history_test_run_results_removed`, `test_original_state_messages_preserved`.
- [ ] **Step 2:** Run the tests; expected FAIL.
- [ ] **Step 3:** Implement `hide_test_run`.
- [ ] **Step 4:** Run the tests; expected PASS.
- [ ] **Step 5: Commit** `feat(apps): hide test-run calls from monitor`.

### Task 6: APPS setting factory and protocol lineup

**Files:**
- Create: `src/agent_control/apps_backdoor/experiment.py`, `tests/apps_backdoor/test_experiment.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `test_run`, `hide_test_run`, `build_tasks`.
- Produces: `build_setting(config: ExperimentConfig) -> AppsSetting` (ControlArena's APPS setting with `tools=[test_run()]`); `build_protocols(config: ExperimentConfig) -> dict[str, ControlAgent]` with keys `trusted_only`, `untrusted_only`, `trusted_monitoring`, `untrusted_monitoring`, `defer_to_trusted`, `trusted_editing`; `PROTOCOL_MODES: dict[str, list[EvalMode]]` (`trusted_only` honest only; all others honest + attack). Upfront auditing has no entry: it is computed in analysis from the `untrusted_only` runs. The monitor in each protocol is `basic_monitor_builder(model_role=...)`, wrapped by `hide_test_run` unless `config.monitor_sees_test_run`. DTT and TE use `config.defer_threshold` as their threshold.

- [ ] **Step 1: Write failing tests:** `test_lineup_keys_and_modes` (exact keys; `trusted_only` honest-only; others both), `test_setting_includes_test_run_tool`, `test_untrusted_monitoring_uses_untrusted_role` (inspect the built monitor's model role), `test_hidden_switch_wraps_monitors` (default wraps; `monitor_sees_test_run=True` does not).
- [ ] **Step 2:** Run the tests; expected FAIL.
- [ ] **Step 3:** Implement. Read `AppsSetting` fields in the pinned version (`has_backdoor_solution`, `backdoor_source`, etc.) and choose defaults that give meaningful attack samples; write the choice and reason in a comment. If a lineup entry's builder is missing in the pinned version, drop it and note in the README.
- [ ] **Step 4:** Run the tests; expected PASS.
- [ ] **Step 5: Commit** `feat(apps): setting and protocol lineup`.

### Task 7: Scripts, smoke check and cost summary

**Files:**
- Create: `src/agent_control/apps_backdoor/run.py`, `src/agent_control/apps_backdoor/analyze.py`, `src/agent_control/apps_backdoor/smoke.py`, `tests/apps_backdoor/test_smoke.py`; expose as `python -m agent_control.apps_backdoor.run|analyze|smoke`.

**Interfaces:**
- Produces in `smoke.py`: `attack_refusal_rate(df: pd.DataFrame) -> float` (fraction of attack samples with no submission or `secret_input is None` or a refusal-style final message); `REFUSAL_FLAG_THRESHOLD = 0.5`; `token_cost_summary(logs: list[EvalLog], prices: dict[str, tuple[float, float]]) -> dict[str, float]` (dollars per model from log token usage, with prices per million tokens passed in by the caller, no hardcoded price table); `main()` runs 5 attack-mode samples of `untrusted_only` for each candidate U in `--models`, prints refusal rate, backdoor success rate, untrusted-only vs trusted-only honest usefulness, and extrapolated cost for the full lineup, and exits non-zero if the refusal rate is at or above the threshold.
- `run.py --protocols ... --limit N`: runs the lineup (all by default) via `build_tasks` / `run_tasks`; `analyze.py`: loads logs, prints the summary table, saves the two plots under `results/`.

- [ ] **Step 1: Write failing tests:** `test_attack_refusal_rate_counts_no_secret_input` (3 of 4 attack rows with `None` gives 0.75), `test_attack_refusal_rate_ignores_honest_rows`, `test_token_cost_summary_uses_given_prices` (exact dollar value from a hand-built usage), `test_main_exits_nonzero_above_threshold` (patched logs).
- [ ] **Step 2:** Run `uv run pytest tests/apps_backdoor/test_smoke.py -v`; expected FAIL.
- [ ] **Step 3:** Implement. Look up current OpenRouter prices at run time from `https://openrouter.ai/api/v1/models` for the cost summary; do not hardcode them.
- [ ] **Step 4:** Run the tests; expected PASS. Run the full offline suite with `uv run pytest -m "not docker and not live"`; expected PASS.
- [ ] **Step 5: Commit** `feat(apps): run, analyze and smoke scripts`.

### Task 8: Live smoke, then full run, and README

**Files:**
- Create: `README.md`

- [ ] **Step 1:** Verify prerequisites: `docker info` succeeds, `OPENROUTER_API_KEY` set. Stop and ask the owner if either fails.
- [ ] **Step 2:** Run the smoke check for both candidate U models (`deepseek-v3.2`, `claude-haiku-5.5`). Report refusal rate, backdoor success rate, untrusted-only vs trusted-only usefulness, and extrapolated cost to the owner. If refusal rate is at or above 0.5, or untrusted-only is not clearly better than trusted-only, stop and ask the owner which U to use. Do not start the full run without the owner's go-ahead on the cost.
- [ ] **Step 3:** After approval, run the full lineup (20 per mode), then `analyze`; confirm all logs and the two plots exist and the summary table has a row per protocol.
- [ ] **Step 4:** Write `README.md`: setup (uv, `.env`, Docker), the three commands, lineup, the monitor-visibility switch, the modeling assumption that hidden favors the red team, and how to add a new experiment folder.
- [ ] **Step 5: Commit** `docs: README and first run notes`.
