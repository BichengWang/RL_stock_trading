# <Report title: whose claim, what window>

> One paragraph, first thing in the file: the verdicts in one sentence each,
> the level each rests on, and every step of the protocol this report
> skipped or did after the fact (for example "pre-registration written after
> the runs: post hoc"). Wording stays neutral.

Pre-registration: [`preregistration.md`](preregistration.md), committed
`<commit sha>` on `<date>`. Authors notified `<date>`; reply:
[`reply.md`](reply.md).

## Summary

| Claim | Level | Claimed | Ours (95% interval) | Verdict |
| --- | --- | --- | --- | --- |

Generated from [`claims.json`](claims.json). What holds, what does not, and the
one thing that would change a verdict.

## Claims

Each claim quoted verbatim with its table or page, the metric, the window,
the baseline, and the threshold that decides it (copied from the
pre-registration).

## Method

- **Level and setup**: L1 / L2 / L3; what ran, with software commits.
- **Data**: source, window, universe (survivorship-free or labeled), SHA-256 of
  every input. No vendor prices are redistributed.
- **Execution and costs**: signal at the close, fill at the next open, fee per side.
- **Seeds and trials**: number of seeds, configurations tried by us and, if known, by the authors.
- **Smallest detectable effect**: the standard error of an annualised Sharpe
  ratio over this window, by bootstrap, and what that means for each claim.
- **Planned analyses** and the statistics used (seed-averaged returns, paired
  stationary bootstrap, deflated Sharpe, multiple-comparison note).

## Results

Tables from the evidence files, each number citing the file and record it
came from; per-seed results always included.

## Verdicts

One subsection per claim: the rule, the numbers, the verdict, the reasoning.

## Deviations

Everything that differs from the pre-registration or from the authors' setup,
with the reason. Analyses added after the first run are listed here.

## Limitations

What the test cannot say.

## Corrections

Dated list of errors found after publication, with credit to whoever found
them. Empty until then.

## Reproduce

Exact commands, in order, from a clean checkout.
