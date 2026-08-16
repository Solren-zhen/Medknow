## Summary

<!-- One or two sentences: what does this PR change and why? -->

## Type of change

- [ ] Bug fix
- [ ] New uncertainty method
- [ ] New dataset / external cohort
- [ ] Leaderboard entry / result update
- [ ] Documentation
- [ ] Other (describe)

## Honesty checklist

MedKnow requires that every number is real and every claim is run. Please confirm:

- [ ] All new metrics come from actual runs (no fabricated numbers).
- [ ] Code that is ready but **not** run is marked explicitly as not-yet-run.
- [ ] Protocol choices (ECE style, inference mode, temperature) are documented
      in `configs/` `evaluation.protocols` if they differ from existing ones.
- [ ] `pytest tests/ -v` passes locally.
- [ ] `ruff check src/medknow tests scripts` is clean.
- [ ] README / CHANGELOG updated if behavior or numbers changed.

## How to verify

<!-- Exact commands a reviewer can run to reproduce your results. -->

```bash
# e.g.
pytest tests/ -v
python scripts/medknow_compare_referral_methods.py
```
