# Public 360 study — Tier 1 and Tier 2 results (post hoc, 2026-10-01)

Source: `runs/public-tier12-20261001/` on Vulcano. Tables are generated from
`tier1/*.json` and from `{development,confirmation}/<key>/summary.json`. The run
of record `public-submission-full-gpu-20260930` was opened read-only; its 1,227
files are byte-identical before and after (`original_manifest_verify.json`,
04:20:11Z) and again after the F3 supplement
(`original_manifest_verify-20261001T115942Z.json`, 11:59:42Z).

## Tier 1 — saved scores, no refits

All intervals below come from saved scores. The fitted nuisances are held fixed,
as in the primary interval.

- **Headline reproduced exactly.** ATT 0.0307738, SE 0.0120398, G = 64.
- **Dependence across teams** (`tier1/dependence.json`):

  | clustering | SE | analytic 95% CI | wild cluster bootstrap 95% CI (Webb, B = 9,999) | p (ATT = 0) |
  |---|---|---|---|---|
  | match (primary) | 0.0120 | [0.0067, 0.0548] | [0.0072, 0.0546] | 0.013 / WCB 0.010 |
  | team, G = 32 | 0.0131 | [0.0041, 0.0574] | [0.0042, 0.0576] | 0.025 / 0.023 |
  | opponent, G = 32 | 0.0165 | [−0.0028, 0.0643] | [−0.0020, 0.0634] | 0.071 / 0.068 |
  | two-way team × opponent (CGM) | 0.0162 | [−0.0022, 0.0637] | — | 0.066 |

- **Null offset diagnosed** (`tier1/null_offset.json`).
  - In the within-match permutation nulls, the fitted ATT exceeds the
    known-assignment ATT by +0.00187 on average (MCSE 0.00003). The offset is
    nearly constant across the 50 replicates (SD 0.00022).
  - A plug-in for the between-match component predicts +0.00211.
  - Mechanism: permutation keeps each match's shot share, and the nuisances have
    no match term.
- **Real-data counterpart (exploratory).** A match-stratified (within-match
  Hajek) reweighting of the saved nuisances gives 0.0307, CI [0.0072, 0.0542].
  Between-match comparisons do not drive the estimate.
- **Status relabel** (`tier1/status_relabel.json`). `completed` means all
  pre-declared computations completed. It is not a validation of identification.
- **Robustness table.** `tier1/robustness_with_diagnostics.csv` adds a
  diagnostics-passed column. Four variations fail the diagnostics: carry
  exclusion, local-pressure ablation, removal of match 3857297, and removal of
  team 799.

## Tier 2 — richer covariates (prespecified post hoc; WC = post-hoc exploratory)

The definitions are in `docs/public_360_tier2_prespecification.md` (sha256
`c140f2aa…`), recorded at 04:12:52Z, before the first fit at 04:13:40Z.

- **Group A:** 19 strictly historical event-context features.
- **Group B:** 3 features of 360 geometry (triangle blockers, keeper position),
  computed on the observable sample V.

### Sanity checks

- The baseline refits reproduce the saved estimates: Euro 0.0177434 vs 0.0177437;
  World Cup 0.0307662 vs 0.0307738 (|Δ| = 7.7×10⁻⁶).
- Incoming-pass linkage:

  | Cohort | Arm | Linked | Unresolved | No incoming pass |
  |---|---|---|---|---|
  | Euro | Shots | 79.5% | 7.9% | 12.6% |
  | Euro | Continuations | 88.5% | 3.3% | 8.2% |
  | World Cup | Shots | 76.6% | 8.7% | 14.7% |
  | World Cup | Continuations | 86.5% | 3.4% | 10.0% |

- Membership of the observable sample V depends on the action. Supported shots
  in V: Euro 79%, World Cup 82%. Supported controls in V: Euro 40%, World Cup 40%.

### Identical-sample adjustment changes (F1, F2)

Same supported decisions within each comparison; every arm, baseline included,
gets fresh nuisances.

| cohort | comparison | arm | ATT | 95% CI | control ESS | top-1% weight share | union key / other SMD | gates (own features) | propensity AUC |
|---|---|---|---|---|---|---|---|---|---|
| Euro | F1 (2,195 shots) | baseline | 0.0177 | [−0.0022, 0.0376] | 1,737 | 0.34 | 0.045 / 0.425 | pass | 0.954 |
| Euro | F1 | + A | 0.0133 | [−0.0079, 0.0346] | 1,003 | 0.47 | 0.109 / 0.187 | fail | 0.975 |
| Euro | F1 | Δ(+A) | −0.0044 | [−0.0150, 0.0062] | | | | | |
| Euro | F2 (V, 1,740 shots) | baseline | 0.0240 | [0.0031, 0.0448] | 1,154 | 0.26 | 0.083 / 0.528 | pass | 0.938 |
| Euro | F2 | + A | 0.0163 | [−0.0085, 0.0412] | 671 | 0.36 | 0.130 / 0.526 | fail | 0.967 |
| Euro | F2 | + A + B | 0.0224 | [−0.0037, 0.0485] | 504 | 0.42 | 0.165 / 0.183 | fail | 0.974 |
| Euro | F2 | Δ(+A) / Δ(+A+B) | −0.0076 / −0.0015 | [−0.0223, 0.0071] / [−0.0182, 0.0152] | | | | | |
| **WC** | F1 (1,228 shots) | baseline | 0.0308 | [0.0067, 0.0548] | 1,006 | 0.35 | 0.075 / 0.384 | pass | 0.951 |
| **WC** | F1 | + A | 0.0297 | [0.0076, 0.0519] | 561 | 0.47 | 0.174 / 0.156 | fail | 0.974 |
| **WC** | F1 | Δ(+A) | −0.0010 | [−0.0130, 0.0109] | | | | | |
| **WC** | F2 (V, 1,006 shots) | baseline | 0.0352 | [0.0037, 0.0667] | 669 | 0.28 | 0.066 / 0.529 | pass | 0.937 |
| **WC** | F2 | + A | 0.0337 | [0.0076, 0.0598] | 364 | 0.37 | 0.188 / 0.595 | fail | 0.967 |
| **WC** | F2 | + A + B | 0.0357 | [0.0038, 0.0677] | 250 | 0.46 | 0.302 / 0.227 | fail | 0.975 |
| **WC** | F2 | Δ(+A) / Δ(+A+B) | −0.0015 / +0.0005 | [−0.0176, 0.0146] / [−0.0274, 0.0285] | | | | | |

Notes on the table:

- The union SMD is computed on all features of the comparison. For baseline
  arms it is therefore large, because those arms do not adjust for the new
  features.
- Opponent-clustered and two-way SEs for every arm are in
  `report/tier2_estimates.csv`. For example, the World Cup F1 +A arm has
  opponent SE 0.0153 and two-way SE 0.0172.

### Re-selected support population (F3, reported separately)

The Euro design was rebuilt with Group A in both the model features and the
support features.

- **Selected models:** HGB propensity and ridge m0. The one-SE CUDA rule did not
  select XGBoost here.
- **Support threshold:** no threshold passed the gates, so selection fell back
  to 0.05 (`selection_passed: false`).
- **Euro (all-Euro out-of-fold, not held out):** 0.0115, CI [−0.0113, 0.0343];
  2,186 shots; fails balance. Shot retention against the prespecified P0-supported
  denominator is 2,186 / 2,195 = 0.996; overlap (Jaccard) with the primary
  supported shots is 0.996. Both come from
  `development/6c15146a1e7f/f3_supplement.json`, computed from saved artifacts
  without a refit; the saved F3 ATT is reproduced exactly (difference 0.0).
- **WC:** 0.0292, CI [0.0081, 0.0504]; 1,223 shots; overlap with the primary
  supported shots (Jaccard) 0.996; ESS 564; fails balance and donor
  concentration.
- **Superseded Euro F3 retention.** `report/tier2_estimates.csv` and the Euro
  `summary.json` carry the earlier shot retention 0.943 (2,186 / 2,317; the
  denominator was all eligible treated rows of the re-selected sample, not the
  P0-supported shots). Those files were not overwritten. Use 0.996 from
  `f3_supplement.json`; the World Cup row (1,223 / 1,228 = 0.996) was already on
  the prespecified basis.

## Reading (per the prespecified interpretation rules)

- **World Cup (exploratory).** Adding Group A changes the estimate by −0.0010
  (paired CI [−0.0130, 0.0109]). On the observable sample, Groups A + B change it
  by +0.0005. This is robustness only to the tested additions.
- **Euro.** The shifts are larger but imprecise: Δ(+A) = −0.0044 (F1) and −0.0076
  (F2), and the paired CIs include zero. The F1 baseline CI includes zero. The F2
  baseline CI [0.0031, 0.0448] excludes zero, while the richer F2 arms and F3
  include it. Under the prespecified rules, a smaller point estimate may reflect
  additional adjustment, a different selected population or measurement of the
  new features. It is not evidence of confounding of a given size.
- **Every richer arm fails the pre-declared key and other balance gates.**
  - Euro (+A, +A+B, F3): these are the only gates that fail. Donor and match
    concentration pass.
  - World Cup (F1 +A, F2 +A, F2 +A+B, F3): donor concentration also fails. F2
    +A+B also fails the effective-donors gate.
  - The new features are strongly predictive of shooting (AUC 0.95 → 0.975), so
    the weights become more concentrated and the ESS falls by 42–63%. These are
    therefore sensitivity estimates under weaker overlap, not better-identified
    replacements for the primary.
- **Group B carries known caveats.** Its frame is captured at the event, a
  post-treatment risk. Membership in V depends on the action (about 80% of
  shots vs 40% of controls).
- **Team dependence remains the main inferential limitation.** Under opponent or
  two-way clustering, the primary interval includes zero (p ≈ 0.07).
