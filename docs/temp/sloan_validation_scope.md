# Sloan precedent, simulation datasets and bootstrap allocation

Read-only research follow-up, 2026-09-30. No experiment, approved configuration,
running job or GPU allocation changed. This clarifies and revises recommendations
in `public_360_compute_audit.md`, not the implemented validation gates.

## Three different sample counts

The only real empirical competitions remain Euro 2020 and Euro 2024 for development,
and World Cup 2022 for held-out confirmation. The World Cup is a held-out cohort,
not chronologically later than Euro 2024.

The simulation's 200 datasets are **200 independently generated artificial datasets
per scenario**, not 200 tournaments to acquire. Each currently has 48 imaginary
matches × 250 decisions, generated with known potential-outcome expectations.
Varying signs, overlap and misspecification tests whether the pipeline recovers the
known effect for its selected supported treated population. Simulated data are
method-validation inputs, never substitutes for empirical football findings.

One bootstrap replicate/draw resamples whole original matches with replacement
from a fixed empirical or synthetic dataset and refits the estimator. Bootstrap is
the procedure; a draw/replicate is one iteration. The inference draw returns a new
ATT, not a new tournament or new independently observed matches. Duplicate matches
retain their original grouping for training/evaluation exclusion.

Thus R synthetic datasets × B bootstrap draws within each dataset are separate
dimensions, not repetitions of the same kind. The existing six-case grid uses
6 × R=200 × B=1,000, producing 1.2 million pipeline refits before the two stress
cases. Neither count is inherently required by Sloan.

## Primary-paper comparison

This is a targeted sample of award-winning/finalist manuscripts, not an exhaustive
survey or proof about unpublished experiments. Author institutions confirm award
status; the official Sloan previous-finalist list identifies accepted presentation
papers. Methods below were read in the full PDFs, not inferred from abstracts.

### 2025 winner: penalty-kick location optimization

Full [conference paper](https://cdn.prod.website-files.com/68d6be744d7efccc2207f571/68d6be744d7efccc220808ea_Do%20Behavioral%20Considerations%20Cloud%20Penalty-Kick%20Location%20Optimization%20in%20Professional%20Soccer.pdf),
award confirmed by [Syracuse](https://falk.syr.edu/the-ultimate-goal/).

Page 2: 536 UEFA penalty kicks from 2015–2020, eight goal regions.
Empirical evidence combines polynomial regressions, spatial displays and tuned
Lasso robustness. Pages 5–7 report one billion simple game-theoretic payoff-pair
draws and 100 samples of 100 game trials. Page 9 reports 30 correlation resamples
per subset; page 11 reports bootstrap Lasso standard errors without a stated draw
count. These cheap simulations are not billion learned-pipeline refits. No
analogous full-procedure nested confidence-interval coverage grid is reported.
The webpage title mentions gradient boosting, but this full PDF describes Lasso;
the comparison follows the PDF rather than assuming methods from the title.

### 2024 winner: fair darts handicaps

Full [conference paper](https://cdn.prod.website-files.com/68d6be744d7efccc2207f571/68d6be744d7efccc22080818_193939%20-%20No%20More%20Throwing%20Darts%20at%20the%20Wall-%20Developing%20Fair%20Handicaps%20for%20Darts.pdf),
award confirmed by [University of Toronto](https://news.engineering.utoronto.ca/u-of-t-engineering-research-wins-first-place-at-mit-sloan-sports-analytics-conference/).

Section 2 uses previously fitted skill distributions based on 16 professionals;
section 4.4.3 mainly uses Michael Smith as a representative baseline with scaled
execution error. The MDP has >1.8 million states, 729 aiming targets and 62 credit
actions, solved with structured dynamic programming. Section 7 and Appendix B
simulate 10,000 games per player pairing to compare heuristic and optimized
head-start/dynamic-credit policies. This is substantial computation, but game
simulation evaluates solved policies without retraining the whole skill model
each time. No nested bootstrap coverage grid is reported.

### 2023 runner-up: optimal pitch sequencing

Full [conference paper](https://cdn.prod.website-files.com/68d6be744d7efccc2207f571/68d6be744d7efccc22080794_A%20Game%20Theoretical%20Approach%20to%20Optimal%20Pitch%20Sequencing.pdf),
second-place status recorded by [coauthor David Grimsman](https://www.davidgrimsman.com/publications).

Section 4.1 (pp.11–13) uses MLB 2021–2022 Statcast data, a 75%/25% plate-appearance
training/validation split, nine outcome calibration plots, and a memoryless-model
ablation. Sections 7–8 examine equilibrium choices and real at-bat examples;
section 8.1 narrows the expensive decision-point analysis to ten batter/pitcher
matchups and 255 pitches. No analogous nested-bootstrap CI-coverage grid is
reported. Scope is tailored to the proposed model/strategic application.

### 2025 causal-inference finalist: CLIPP

Full [conference paper](https://cdn.prod.website-files.com/68d6be744d7efccc2207f571/68d6be744d7efccc220808e9_CLIPP.pdf),
finalist status in [Sloan's official list](https://www.sloansportsconference.com/research-paper-competition).

NFL Big Data Bowl 2023 tracking, doubly robust estimation, engineered adjustment
features. Pages 4–5 report balance, randomized placebo treatment, temporal/contextual
subsamples, calibration/ROC, and sensitivity scenarios omitting/partially observing
confounders or changing treatment definitions. Pages 5–12 include play examples,
player/team rankings, predictive comparisons and external PFF ranking correlation.
The manuscript does not specify its analysed N, exact split proportions, match-grouped
cross-fitting details, bootstrap count, number of simulated datasets or a formal
CI-coverage grid. It does mention sensitivity simulations; do not claim none occurred.
Its acceptance neither proves exchangeability nor justifies copying its methodological
omissions. It is useful precedent for a focused submission, not a validation standard.

### Related soccer finalist: 2019 EPV framework

Full [conference paper](https://cdn.prod.website-files.com/68d6be744d7efccc2207f571/68d6be744d7efccc2207fab6_Decomposing-the-Immeasurable-Sport.pdf)
and official [paper page](https://www.sloansportsconference.com/research-papers/decomposing-the-immeasurable-sport-a-deep-learning-expected-possession-value-framework-for-soccer).
Page 5 describes EPL 2012–2013 and Barcelona 2017–2018/2018–2019 optical tracking.
Component models are described as tuned/calibrated; the paper explicitly prioritizes
interpretation/applications over technical detail. Sections 4–6 emphasize contextual
passing, off-ball value and visual decision analysis. No full-procedure causal-ATT
coverage grid is reported. Predictive possession valuation is a different claim
from identification of a shooting ATT, so its validation cannot simply be copied.

## Interpretation for our submission

The official [evaluation criteria](https://www.sloansportsconference.com/research-paper-competition)
emphasize novelty, appropriate rigor, reproducibility, application and impact; no
numeric resampling quota is stated. The live page mixes current deadlines with
older rule wording, so do not attribute every paragraph to one historical edition.

The examples are compatible with focused, claim-specific evidence, not a mandatory
exhaustive inference-method benchmark. Some are computationally large in other
ways, which makes iteration count alone a poor cost measure.

Our central contribution is an applied supported shooting-versus-continuation
comparison, with honest coverage limitations and practical tactical insight—not
necessarily a new universally validated inference algorithm. Keep defensible support,
timing, match-excluded nuisance predictions, appropriate uncertainty, software nulls,
quantified hidden-confounding sensitivity and untouched-cohort assessment. Retain a
bounded known-truth simulation targeting our particular risks, but do not impose
1.6 million refits as a Sloan admission threshold. Winning/acceptance does not excuse
unjustified causal claims; unresolved issues should narrow those claims.

## Is B=1,000 necessary?

No universal threshold exists. Bootstrap draw count controls numerical Monte Carlo
precision in approximating a chosen bootstrap distribution. It does not increase
the real-match sample size, establish exchangeability or fix an invalid resampling
scheme/adaptive-support estimator.

The current interval takes the 2.5th and 97.5th percentiles of the resampled ATTs.
Expected observations in each 2.5% tail are 1.25 at B=50, 5 at B=200, 12.5 at
B=500 and 25 at B=1,000. Fewer draws make endpoints more sensitive to random seed.
This arithmetic is a precision diagnostic, not a validity theorem or guaranteed
minimum. Heavy/skewed tails can require more.

[Hesterberg's methodological treatment](https://arxiv.org/html/1411.5279#S3.SS6)
distinguishes standard-error estimation from tail/CI accuracy and explicitly treats
draw count as Monte Carlo precision. It discusses the historical 200-for-SE /
1,000-for-CI suggestions and recommends substantially more for stringent precision
when computationally affordable. Therefore it cannot be cited as guaranteeing
500 is enough for our procedure—or 1,000 sufficient in every application.

Revised proposal, pending implementation/design approval:

- 200 draws for development feedback, 500 as a provisional final empirical target.
- Measure and report Monte Carlo uncertainty in interval endpoints/SE; saved ATT
  numbers can be resampled cheaply for this diagnostic, without new model fits.
- Predeclare precision tolerances and batch/cap rules before looking at whether
  zero is excluded. Expand toward 1,000 or report inadequate numerical precision
  if necessary; never stop because the effect becomes significant.
- Do not require 1,000 draws inside every synthetic dataset, null or secondary
  analysis. Match its uncertainty allocation to its actual claim.
- Retain all finite extreme or diagnostically poor draws; disclose genuine failed
  attempts. 'Successful' means a computation completed, not a favorable estimate.

This revises the earlier blanket recommendation to retain 1,000. It is a transparent
compute/precision tradeoff, not an assurance that B=500 is sufficient or a change
already applied to the running experiment.
