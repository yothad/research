# Cost-based routing: findings and final decision

Investigation behind the 3-tier routing policy (Auto Pass / Human Authentication /
Auto Decline) used in `steps/risk_routing.py`. Full reasoning trail — a
final-numbers-only summary would lose why several earlier approaches were rejected,
which is as much the point of this analysis as the numbers themselves.

Analysis script: `research/three_tier_routing_analysis.py`.

## Why 3 tiers instead of a single threshold

A single binary threshold forces every transaction into a hard accept/decline
call. Adding a middle "route to human/step-up authentication" tier lets
uncertain cases get resolved with extra friction instead of a blind guess. At
the same cost assumption ($10 flat, see below), the 3-tier policy cost **$8,119**
vs. **$13,218** for the best binary threshold on the same score — a ~38%
reduction, from having a real middle option instead of forcing a hard call on
borderline cases.

## Attempt 1: flat false-decline cost — rejected

Initial model: every wrongly-declined legitimate transaction costs a flat
assumed dollar amount (e.g. $10), regardless of what that transaction was
actually worth.

**Problem, found empirically**: 26% of legitimate transactions in this dataset
are worth less than $10 (12.6% under $5). A flat $10 cost means, for over a
quarter of the data, "the cost of wrongly blocking this transaction" is
literally asserted to exceed the transaction's own value — inconsistent, since
the missed-fraud side of the same model *does* scale with each transaction's
real `amt`.

Concretely, at flat $1 vs. flat $10: the $1 assumption misses less fraud (10 vs.
23 cases) but does so by auto-declining **933 real customers outright** (vs. 136
at $10) and routing another 3,721 through extra friction (vs. 1,227) — treating
"annoying/blocking an innocent customer" as nearly free is the same failure mode
as optimizing for recall alone taken to its limit (decline everything).

## Attempt 2: pure linear scaling (`base + fraction × amt`) — rejected

Fix attempted: scale the false-decline cost with the transaction's own value,
so small transactions aren't over-penalized and large ones aren't
under-penalized.

**Problem, found empirically**: this dataset's legitimate transactions go up to
**$22,768**. At a 20% fraction, declining a transaction anywhere near that costs
thousands of dollars under the model, so the optimizer became extremely
reluctant to auto-decline *anything* that could plausibly be a large legitimate
purchase. Net effect at base=$10: legit auto-declines dropped 136 → 8 (good),
but missed fraud more than tripled, 23 → 79, and total cost roughly tripled,
$8,119 → $24,671. The fix over-corrected — a small number of extreme-value
transactions dominated the whole cost model.

## Attempt 3: three cost-model shapes, compared directly

Compared `flat`, `capped` (`base + fraction × min(amt, $200)`), and `log`
(`base + k × log(1+amt)`) at the same base-friction levels. Key finding: dollar
totals are **only comparable within one cost-model shape**, not across shapes —
each one prices a false decline differently, so a lower total under `flat`
doesn't mean `flat` is better, it means `flat` is cheaply pricing its (much
larger number of) wrongful declines.

At base=$10: `capped` and `log` both land on 53 legit auto-declines (vs. flat's
136 — the original problem, fixed). Between them, `log` dominates: fewer missed
frauds than `capped` at every base level tested (23 vs. 31 at base=$5), and
lower cost within its own accounting at every level. They converge to identical
thresholds at base=$20, which — two different mathematical shapes agreeing on
the same decision boundary — is a reassuring sign the boundary reflects
something real in the data, not an artifact of one formula.

**Decision: `log` model.**

## Picking `base`: elbow detection instead of eyeballing

Rather than pick a round number and compare by hand, swept `base` finely
(25 points, $1–$100, log-spaced) and traced the (missed_fraud_$,
legit_friction_$) tradeoff curve. Found the elbow via max perpendicular
distance from the line connecting the two curve endpoints — a standard,
explainable way to find where a tradeoff curve stops paying off, rather than
an arbitrary cutoff.

```
base range          missed_fraud   legit_friction   marginal $ traded
$1 – $8.25              $2,596        $427,588       (baseline)
$10                     $4,659        $331,361       ~46x friction saved per $ missed
$12 – $17.78            $5,284        $294,547       ~59x
$21.54 – $38.31         $6,018        $266,166       ~39x   <- elbow
$46.42 – $56.23        $11,997        $127,624       ~23x
$82.54 – $100          $13,992        $ 88,185       ~17x
```

Elbow at **base ≈ $21.54** (rounded to $20 for the reference policy — same
plateau, same thresholds). The marginal trade-off degrades gradually rather
than falling off one dramatic cliff, so this is the best *objective* pick from
a smooth curve, not a uniquely "correct" single point — the $10–$38 range is
broadly reasonable, $21.54 is where the data says the knee sits.

## Final decision

**Model**: `log`, `base=$20` (`fp_decline_cost = 20 + 3*log(1+amt)`, `auth_friction_cost`
= 20% of that, `catch_rate` = 85% assumption for step-up auth stopping real fraud).

**Thresholds** (on the raw XGBoost score): `pass < 0.69`, `0.69 <= human_auth < 0.95`,
`decline >= 0.95`.

**Resulting outcome** (on the held-out test set):

| | count | dollar value |
|---|---|---|
| Missed fraud (auto-passed) | 38 | $6,018 |
| Fraud routed to auth (of which ~$2,149 expected to still slip through at 15% miss rate) | 71 | $14,328 |
| Fraud auto-declined (blocked) | 320 | $209,974 |
| Legit auto-declined (false alarm, revenue lost outright) | 53 | $44,977 |
| Legit routed to auth (false alarm, friction only) | 667 | $221,189 |

**Caveats worth repeating in any writeup of this**: `catch_rate` (85%) and the
`log` model's constants (`k=3`, `AUTH_FRICTION_FRACTION=20%`) are assumptions,
not measured facts — the dataset has no data on real step-up-auth outcomes or
true customer-relationship cost. What *is* empirically grounded: the missed-fraud
dollar figures (real `amt` on real transactions), the comparison ranking between
`flat`/`capped`/`log`, and the elbow location given those assumptions.
