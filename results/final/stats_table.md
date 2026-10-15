# Statistics table

Source of every value: stats_table.csv (column `source`). Friedman on the 10 folds x 10 models matrix of per-fold test ROC-AUC; Nemenyi alpha 0.05; Nadeau-Bengio corrected resampled t-test of the model with the highest mean fold AUC vs each other model, Holm-adjusted over 9 comparisons.

## D01_child

Friedman chi2 = 7.69, p = 0.566; Iman-Davenport F = 0.84, p = 0.581; Nemenyi CD = 4.28; significant Nemenyi pairs: 0 of 45.

Average ranks (1 = best): M04 4.00, M07 4.75, M05 5.20, M08 5.30, M06 5.35, M01 5.45, M09 5.75, M10 6.00, M03 6.45, M02 6.75.

| comparison | mean AUC best | mean AUC other | mean diff | t | p | Holm p |
|---|---|---|---|---|---|---|
| M04 vs M01 | 0.856 | 0.761 | +0.094 | 1.03 | 0.330 | 1.000 |
| M04 vs M02 | 0.856 | 0.678 | +0.178 | 1.34 | 0.214 | 1.000 |
| M04 vs M03 | 0.856 | 0.744 | +0.111 | 1.75 | 0.115 | 1.000 |
| M04 vs M05 | 0.856 | 0.767 | +0.089 | 1.04 | 0.324 | 1.000 |
| M04 vs M06 | 0.856 | 0.783 | +0.072 | 1.09 | 0.302 | 1.000 |
| M04 vs M07 | 0.856 | 0.800 | +0.056 | 0.69 | 0.509 | 1.000 |
| M04 vs M08 | 0.856 | 0.828 | +0.028 | 0.24 | 0.814 | 1.000 |
| M04 vs M09 | 0.856 | 0.706 | +0.150 | 1.22 | 0.253 | 1.000 |
| M04 vs M10 | 0.856 | 0.761 | +0.094 | 1.39 | 0.199 | 1.000 |

## D01_clip

Friedman chi2 = 21.19, p = 0.012; Iman-Davenport F = 2.77, p = 0.007; Nemenyi CD = 4.28; significant Nemenyi pairs: 0 of 45.

Average ranks (1 = best): M07 3.30, M04 3.90, M08 4.10, M05 4.60, M01 5.40, M06 6.10, M10 6.30, M03 6.50, M09 7.30, M02 7.50.

| comparison | mean AUC best | mean AUC other | mean diff | t | p | Holm p |
|---|---|---|---|---|---|---|
| M08 vs M01 | 0.686 | 0.645 | +0.041 | 1.25 | 0.244 | 1.000 |
| M08 vs M02 | 0.686 | 0.595 | +0.092 | 1.67 | 0.128 | 1.000 |
| M08 vs M03 | 0.686 | 0.635 | +0.051 | 1.27 | 0.235 | 1.000 |
| M08 vs M04 | 0.686 | 0.683 | +0.004 | 0.09 | 0.928 | 1.000 |
| M08 vs M05 | 0.686 | 0.657 | +0.029 | 0.64 | 0.539 | 1.000 |
| M08 vs M06 | 0.686 | 0.639 | +0.047 | 1.29 | 0.231 | 1.000 |
| M08 vs M07 | 0.686 | 0.678 | +0.008 | 0.23 | 0.821 | 1.000 |
| M08 vs M09 | 0.686 | 0.605 | +0.081 | 1.58 | 0.148 | 1.000 |
| M08 vs M10 | 0.686 | 0.638 | +0.048 | 1.12 | 0.290 | 1.000 |

## D02_image

Friedman chi2 = 62.92, p = 3.7e-10; Iman-Davenport F = 20.92, p = 9.7e-18; Nemenyi CD = 4.28; significant Nemenyi pairs: 11 of 45 (M02-M04 p 2.5e-06, M02-M06 p 0.027, M02-M07 p 4e-05, M02-M08 p 0.043, M03-M04 p 0.013, M04-M05 p 6.9e-05, M04-M09 p 6.9e-05, M04-M10 p 0.00056, M05-M07 p 0.00077, M07-M09 p 0.00077, M07-M10 p 0.005).

Average ranks (1 = best): M04 1.40, M07 2.10, M06 4.20, M08 4.40, M01 4.80, M03 6.25, M10 7.30, M05 7.90, M09 7.90, M02 8.75.

| comparison | mean AUC best | mean AUC other | mean diff | t | p | Holm p |
|---|---|---|---|---|---|---|
| M04 vs M01 | 0.931 | 0.901 | +0.030 | 4.15 | 0.003 | 0.010 |
| M04 vs M02 | 0.931 | 0.867 | +0.065 | 2.65 | 0.027 | 0.053 |
| M04 vs M03 | 0.931 | 0.890 | +0.041 | 4.94 | 0.0008 | 0.006 |
| M04 vs M05 | 0.931 | 0.881 | +0.050 | 5.65 | 0.00031 | 0.003 |
| M04 vs M06 | 0.931 | 0.901 | +0.030 | 4.82 | 0.00095 | 0.006 |
| M04 vs M07 | 0.931 | 0.924 | +0.007 | 0.86 | 0.413 | 0.413 |
| M04 vs M08 | 0.931 | 0.903 | +0.028 | 3.53 | 0.006 | 0.019 |
| M04 vs M09 | 0.931 | 0.875 | +0.056 | 4.26 | 0.002 | 0.010 |
| M04 vs M10 | 0.931 | 0.885 | +0.046 | 5.10 | 0.00065 | 0.005 |

## Pretrained vs from scratch (descriptive)

| analysis | group | mean of model fold-mean AUC | range | mean pooled AUC |
|---|---|---|---|---|
| D01 child | from scratch (M02,M05,M09) | 0.717 | 0.678-0.767 | 0.586 |
| D01 child | pretrained (M01,M03,M04,M06,M07,M08,M10) | 0.790 | 0.744-0.856 | 0.755 |
| D01 clip | from scratch (M02,M05,M09) | 0.619 | 0.595-0.657 | 0.573 |
| D01 clip | pretrained (M01,M03,M04,M06,M07,M08,M10) | 0.658 | 0.635-0.686 | 0.651 |
| D02 image | from scratch (M02,M05,M09) | 0.874 | 0.867-0.881 | 0.870 |
| D02 image | pretrained (M01,M03,M04,M06,M07,M08,M10) | 0.905 | 0.885-0.931 | 0.884 |
