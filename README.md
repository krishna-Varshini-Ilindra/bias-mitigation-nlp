# Can Score Calibration Fix Gender Bias in a Sentiment Model?

**Author:** Krishna Varshini Ilindra  
**Background:** M.S. in Computer Science, University of Bridgeport · Software Engineer, Octagon IT  
**Run the notebook:** [Open in Colab](https://colab.research.google.com/github/krishna-Varshini-Ilindra/bias-mitigation-nlp/blob/main/bias_mitigation_nlp.ipynb)

---

## The Starting Point

In my [previous project](https://github.com/krishna-Varshini-Ilindra/sentiment-bias-nlp), I measured demographic bias in a RoBERTa sentiment model using template sentences that differ only in a name. On positive sentences, female names received slightly higher positive scores than male names.

The natural next question was: can a gap like this be reduced without retraining the model?

---

## The Approach

I tested two lightweight ideas on the same model (`cardiffnlp/twitter-roberta-base-sentiment`) and the same 4,096 template sentences (8 templates × 32 names × 16 emotion words).

**1. Counterfactual Data Augmentation (CDA)**
I created a "twin" of every sentence by swapping the name for one of the opposite gender.

**2. Post-hoc Score Calibration (equalized means)**
- Compute each gender's mean positive score across all sentences
- Shift female scores down and male scores up by half the difference, so the two overall means match
- Measure the gap again, along with accuracyal.

---

## Results

### Baseline (before any fix)

| Gender | Positive sentences | Negative sentences |
|---|---|---|
| Female | 0.7873 | 0.0109 |
| Male | 0.7597 | 0.0110 |
| **Gap (F − M)** | **+0.0276** | **−0.0001** |

*Note: the baseline here differs from Project 1 (+0.0285) because Project 1's templates had trailing spaces and one capitalized emotion word.*

### Attempt 1 — CDA: no change (0.0%)

CDA is meant to rebalance **training** data, but here the model is not retrained. Because every female name in the dataset already has a male counterpart, swapping names simply reproduced the same 4,096 sentences, so the gap could not change. This was a design mistake, and I've kept it in the notebook to document it.

### Attempt 2 — Score calibration

| Metric | Before | After |
|---|---|---|
| Gap on positive sentences | +0.0276 | +0.0139 |
| Gap on negative sentences | −0.0001 | −0.0130 |
| Total gap (sum of both, absolute) | 0.0277 | 0.0269 |
| Female accuracy | 94.7% | 94.6% |
| Male accuracy | 93.6% | 93.8% |

---

## Visualization

![Bias Mitigation Results](bias_mitigation_results.png)

---

## Key Finding

At first, calibration looked like a success: the gap on positive sentences fell by about half (0.0276 → 0.0139), and accuracy barely changed.

Looking at the full results told a different story. Calibration shifts every female score down and every male score up by the same constant, so it also **created a gap of almost the same size on negative sentences** (−0.0001 → −0.0130). Across both slices, the total gap barely changed (0.0277 → 0.0269). The bias was moved, not removed.

**What I learned:**
- A fairness fix has to be evaluated on **every slice** of the data, not only the slice it was designed for.
- The calibration offset was estimated and evaluated on the **same sentences**. A fair test needs held-out data.
- Group-specific calibration requires knowing each input's demographic group at prediction time, which is often unavailable or inappropriate in practice.
---

## Open Questions

- Would a calibration fitted **per sentiment class** (or per score range), and tested on held-out templates, actually reduce the total gap?
- Do training-time methods (CDA with fine-tuning, or fairness-constrained fine-tuning) reduce the gap without shifting it elsewhere?
- Does the same pattern appear for race and dialect, or for intersectional groups?

---

## Connection to Project 1

This is the second part of a two-project series on fairness in NLP:

- [Project 1 — Measuring Bias](https://github.com/krishna-Varshini-Ilindra/sentiment-bias-nlp): measured race, gender, and dialect gaps in a RoBERTa sentiment model
- **Project 2 — Attempting to Reduce Bias** (this repo): tested CDA and score calibration, and found that calibration shifted the gender gap between slices rather than removing it

---

## How to Run

1. Click **Open in Colab** above
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells top to bottom

---

## References

- Kiritchenko, S. & Mohammad, S. (2018). Examining Gender and Race Bias in Two Hundred Sentiment Analysis Systems. *\*SEM 2018*.
- Zhao, J. et al. (2018). Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods. *NAACL 2018*.
- Barbieri, F. et al. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. *Findings of EMNLP 2020*.
