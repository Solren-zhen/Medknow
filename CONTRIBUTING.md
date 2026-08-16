# Contributing to MedKnow

Thanks for your interest in MedKnow! We welcome contributions of all kinds —
new uncertainty methods, more external cohorts, bug fixes, documentation, and
issue reports. Please read this guide first, and our
[Code of Conduct](CODE_OF_CONDUCT.md).

## The one rule: honesty

MedKnow's entire value is that every number in the README and manuscript is
**reproducible and verifiable**. Therefore:

1. **No fabricated numbers.** Every reported metric must come from an actual run.
2. **No unrun claims.** If code is ready but not executed, mark it explicitly as
   "not yet run" (see the PENDING section in
   [results/tables/manuscript_verification.md](results/tables/manuscript_verification.md)).
3. **Document protocol choices.** If your analysis uses a different evaluation
   protocol (ECE style, inference mode, temperature), record it in
   `configs/` under `evaluation.protocols`.

## Getting started

```bash
# 1. Fork and clone, then install the dev environment
pip install -e ".[dev]"

# 2. Run the test suite (all 74 tests must pass)
pytest tests/ -v

# 3. Lint (must be clean)
ruff check src/medknow tests scripts
```

## How to add a new uncertainty method

This is the most common contribution and the easiest to review:

1. Add a function in `src/medknow/uncertainty/` following the existing pattern
   (e.g. `mc_dropout.py`). All methods share the interface
   `estimate_uncertainty(model, loader, method)`.
2. Register the method name in the config and in
   `scripts/medknow_compare_referral_methods.py`.
3. Add unit tests in `tests/test_uncertainty.py`.
4. Run `python scripts/medknow_compare_referral_methods.py` and update the
   [Leaderboard table](README.md#12-leaderboard--submit-your-method) in the README
   with your real numbers (include the `results/metrics/predictions/*.npz` files).

## How to add a leaderboard entry

The benchmark table in README `12 is the project's growth engine. To add your
method (or a new external cohort):

1. Run the standard pipeline: `scripts/medknow_evaluate_external.py` and
   `scripts/medknow_compare_referral_methods.py`.
2. Include the raw prediction files (`results/metrics/predictions/*.npz`) so the
   numbers are independently checkable.
3. Open a PR that updates the table; maintainers will re-run and verify.

## Pull request process

1. Branch from `main` (`git checkout -b feat/my-change`).
2. Make focused commits; run `pytest` and `ruff` locally.
3. Open a PR using the [template](.github/PULL_REQUEST_TEMPLATE.md).
4. CI runs tests + lint on every PR; it must pass.
5. A maintainer reviews and merges.

## Reporting issues

- **Bugs / unexpected results** → use the
  [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml). Include the
  command you ran, your environment (Python/PyTorch versions), and the exact
  numbers you got.
- **New ideas / features** → use the
  [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml).
- **Security vulnerabilities** → do **not** open a public issue; see
  [SECURITY.md](SECURITY.md).

## Documentation

- Keep the README and [ROADMAP.md](ROADMAP.md) in sync with any behavior change.
- If you change a number anywhere, update
  [results/tables/manuscript_verification.md](results/tables/manuscript_verification.md)
  too.

Thanks for helping make honest medical-AI research reproducible! ⭐