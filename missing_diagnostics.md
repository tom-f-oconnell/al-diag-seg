# Flies without (all of) the diagnostic odors

Presenting the diagnostic panel adds to every experiment, so: can a model work on a fly that lacks some or all of it? Measured
2026-11-15 on the odor response database's fit 7 (86 flies, 20 odors common to all of them) and the database's glomerulus
responses; nothing was retrained, and nothing here is built yet. The scripts are in the database's repository
(`notes/missing_diagnostics/`).

**As it is, no.** `alseg.fly.load_fly` refuses a fly that lacks any of the model's odors. But the approach does not need the
diagnostics as such: it needs some odor responses to find ROIs, and tuning to a panel it knows to name them.

## What a model uses a fly's responses for

1. Two of Cellpose's three input images: the local correlation of tuning between neighbouring pixels, and the first principal
   component of tuning (a basis shared by the training flies, so tied to the model's odors). The third is the average image.
2. Joining the planes' masks into 3D ROIs (`segment.join_planes`: overlap and tuning).
3. The tuning term of a name's cost (`identity.assign_identities`), beside position, shape, atlas overlap and layout.

## Odors withheld from fit 7's model

Its four held-out flies, scored as a prediction is (`Model.evaluate`: per plane, glomeruli drawn on some planes only not counted
against it), given all 20 odors, some of them (the same ones for all three uses; the component image with the others at the
training flies' mean), none (those two images blank, planes joined by overlap as Cellpose does, no tuning term), or the responses
of the fly's other recordings in place of the diagnostics (for the correlation image and the joining; no component image and no
tuning term, as the model knows no tuning of those odors):

| what the model is given | ROIs a fly | found, per plane | right, per plane | found, 3D | right, 3D | named right |
|---|---|---|---|---|---|---|
| all 20 odors | 52 | 57% | 89% | 49% | 77% | 81/119 (68%) |
| 10 of them (3 draws) | 47 | 49% | 85% | 36% | 65% | 55% |
| 5 of them (3 draws) | 41 | 39% | 80% | 27% | 55% | 46% |
| 2 of them (3 draws) | 36 | 34% | 80% | 20% | 50% | 49% |
| all for the ROIs, no tuning in names | 52 | 57% | 89% | 49% | 77% | 58/119 (49%) |
| none for the ROIs, all for their names | 24 | 20% | 69% | 8% | 33% | 26/45 (58%) |
| none at all | 24 | 20% | 69% | 8% | 33% | 20/45 (44%) |
| the fly's other odors for the ROIs, no tuning in names | 45 | 47% | 80% | 32% | 60% | 53/95 (56%) |
| the fly's other odors for joining only | 26 | 20% | 69% | 12% | 42% | 23/50 (46%) |

- **Finding ROIs needs odor responses, not the diagnostic ones.** With none, the model finds about a third as much. The fly's own
  other odors recover most of it, per fly (found per plane, against all 20 diagnostics): kiwi and control (22 odors) 51% against
  53%, validation2 (27) 49% against 53%, megamat (17) 56% against 70%, Sam's 14 natural-mix odors 34% against 50%. It is the
  correlation image that does it, not the joining.
- **Names lose the most**: 49% right without the tuning term, 68% with it.
- **Fewer diagnostics lose steadily**, with no cliff; two are worth nothing to names.
- "None at all" is a floor, not a verdict: the model was trained with those two images, so blank ones are unlike anything it saw.
  A model fit without them was not tried.

## Can a fly's other odors name its glomeruli?

How often a fly's glomeruli (its ROIs named in the database, drawn by hand) are named right by tuning alone: each fly left out in
turn, its glomeruli matched one to one to the other flies' mean tuning of every glomerulus they have.

| flies | diagnostics alone | their other panel alone | both |
|---|---|---|---|
| Tom, megamat (9 flies, 17 odors) | 79% | 80% | 94% |
| Tom, validation2 (6 flies, 22 odors) | 79% | 95% | 96% |
| Tom, kiwi and control (9 flies, 11 odors each) | 70% | 33% and 52% (65% together) | 76% and 79% |
| Sam, natural mixes and ramps (62 flies, about 10 odors a fly) | 72% | 41% | 76% |
| George (6 flies) | 89% | | |

Megamat and validation2 tell glomeruli apart as well as the diagnostics, or better, so flies with those panels could plausibly do
without the diagnostics, given the tuning the database has of them from flies identified with diagnostics. Kiwi, control and
Sam's panel cannot replace them.

## A shorter diagnostic panel

- **Alone**, the diagnostics add up odor by odor (10 at random: 30 to 57% over the datasets; 5: 11 to 25%). The best 10, added one
  at a time on Sam's 62 flies (alpha-terpineol, methyl hexanoate, p-cresol, farnesol, 3-methylthiopropanol, methyl salicylate,
  ethyl lactate, phenylacetaldehyde, ethyl 3-hydroxybutyrate, acetophenone), give 41% there against 63% for the 22 odors nearly
  all of them have, and 49 to 75% on the other datasets against 66 to 81%; little is gained after about 15.
- **Beside megamat**, four diagnostics (p-cresol, farnesol, geosmin, geranyl acetate) give 93%, against 94% with all 26: chosen
  and counted on the same 9 flies, so optimistic. Beside validation2, none adds much (95% to 98%).

## Limits

Four held-out flies for the first table; 6 to 9 flies for megamat and validation2. The second table is of hand-drawn ROIs and
tuning alone: what the odors can tell apart, not what the whole pipeline would score. Naming predicted ROIs by another panel was
not run end to end.

## What it would take

1. **Names from whatever odors the fly has**: each glomerulus' tuning per odor from the database, over all its datasets, and the
   tuning term over the odors the fly shares with it. No refit.
2. **ROIs from whatever odors the fly has**: a fit with the correlation image of each fly's own odors and without the shared
   component image. 2 to 3 hours of GPU.
3. **Missing diagnostics allowed** in a model as it is (the first table). Small, no refit.

1 and 2 together, scored on megamat-only and validation2-only predictions against the current ones, would say whether those
flies can skip the diagnostics.
