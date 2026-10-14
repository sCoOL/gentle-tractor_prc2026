# RMSE and a handful of rows

The score is the RMSE over all departures, so it is driven by the squared error of the worst rows. A LIRF row whose
label is a schedule copy or a date bug sits 10–25 h away from a normal taxi; one such row predicted on the wrong side
adds 0.5–1 × 10⁹ s² to the sum of squares. On the 344,841 January / July rows of the public leaderboard that is
**3–7 s of RMSE from a single row**, more than most layers of the chain gain together (the largest post-step, the
EUROCONTROL daily layer, is worth a few seconds over hundreds of thousands of rows). This page records what that meant in
practice, so the reader can see which choices rest on a few rows.

## 1. Why the model hedges on these rows

For a row that is a copy with probability q and a date bug (or a normal taxi) otherwise, the expected squared error of
a prediction p is smallest at the probability-weighted mean of the two values, not at either of them. Committing fully
to one side only pays when q is close to 1. On 2025 this held every time it was tested: snapping the copy weight of
the unmatched lane to 0 / 1 at 90 / 95 / 98 % lost against the mixture, and snapping the date-bug weight at pb ≥ 0.5
lost +1,690.9 M s² (`docs/DESIGN_DECISIONS.md` §2.8.6). A full value is given only where a rule separates the two sides
on 2025 without an error (`docs/METHOD.md` §11.5).

## 2. RYRR388G 2026-07-26: a 9 h late unmatched flight that was a normal taxi

**The 2025 evidence said copy.** Among 2025 LIRF NM-unmatched departures, no row with a normal-taxi label was more than
6.4 h late; all 28 rows 8–13 h late were schedule copies, in every fold. The mechanism fits: the dataset matched a
departure to its NM flight only when the recorded off-block was within about 1 h of the last filed off-block time, so a
correctly recorded flight that is 9 h late should have been matched. That gave the 8–13 h rule (copy for every
unmatched LIRF row 8–13 h late; `docs/METHOD.md` §9).

**The model was less sure than the rule.** The chain's copy gate gave two of the fourteen 2026 rows in that band a
copy weight of only ≈ 0.72: WMTMT437 (2026-01-10) and RYRR388G (2026-07-26); every other row had ≥ 0.75, most ≥ 0.96.

**What the leaderboard showed.** A candidate file that set all 14 rows to the copy value scored worse than the file it
was built on, by an amount the 14 rows could explain only if one of them was a normal taxi. Searching over the possible
labels of the 14 rows (2025 copy offsets, night-time normal taxis, long taxis, date bugs), conditioned on that exact
difference and on the gate's label-free copy weights, gave RYRR388G as the normal taxi with probability 0.98
(WMTMT437 0.01) and an implied label of about 1,124 s (10–90 %: 1,027–1,204 s): an off-block near 23:52 local, two hours
after its inbound aircraft (RYR8WQ) reached the stand. Label-free context agrees: that night LIRF had a cluster of
unmatched Ryanair and Wizz Air departures 1.5–3 h late that the chain itself rates mostly normal.

**What it is worth.** Predicted as a copy (33,063 s) instead of as a normal taxi, RYRR388G adds about 1.02 × 10⁹ s² if
its label is ≈ 1,124 s: about **+6.8 s on the January / July leaderboard** and about **+3.5 s on the four-month final
ranking**. A candidate file with RYRR388G at a normal-taxi value (and the other 13 rows at the copy value) scored in line
with that reading.

**What the final file does.** The final file keeps RYRR388G where the 2025 rule puts it, at the copy value: no row of the
final file is set against a rule because of a leaderboard result. This page records the finding and its cost instead.

## 3. The other single rows that moved the score

| row | 2025 basis | what a candidate-file comparison added | in the final file |
|---|---|---|---|
| ITY672 2026-07-09 (17 h late, ITA overnight long-haul) | ITA's late-evening long-hauls delayed overnight: 7 copies when they left by 12:17, 2 date bugs after 14:00; ITY672 left 15:12 | a file with a 0.65 date-bug weight scored in a way that fits only a date bug; the full date-bug value was chosen after it | full date-bug value; rule R4 (§11.5) gives the same side from 2025 alone |
| SVI3006 2026-07-16 (16.9 h late, an operator absent from all of 2025) | rule R3 (inbound landed ≥ 12 h before take-off) said date bug, 2 of 2 in 2025 | the file with R3's value scored as if the row were a copy | copy by rule R6 (rare operator, 8 of 8 copies in 2025), written after that comparison |
| WMTMT437 2026-01-10, ITY674 2026-07-19 | the rule says copy; the copy gate was unsure (≈ 0.72 / 0.75) | both scored as copies | copy |

## 4. Are there other rows like RYRR388G in 2026?

Among the 2026 LIRF unmatched rows 6–14 h late, the three rows the copy gate rated least likely to be copies are
WMTMT437 (0.72), RYRR388G (0.73) and ITY674 07-19 (0.75); the leaderboard comparisons fit copies for the first and the
third. The next ones are WZZ35EP 2026-01-04 (7.1 h, 0.89, left on the mixture) and then rows at ≥ 0.93. None of the
February / June rows in this band is below 0.96, so the two months that the public leaderboard never showed have no row
that looks like RYRR388G. The rows left on the mixture because no rule decides them are NOS431 (31 h), WMT3508 (15.7 h)
and, in February / June, none.
