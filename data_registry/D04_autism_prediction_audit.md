# D04_autism_prediction audit

- Folder: `Autism_prediction dataset`
- Detected data type: **tabular**
- Files: 2 | size: 0.09 MB | max depth: 1
- Files by extension: {'.csv': 2}
- Archives (zip/rar): none
- Licence/citation files: none (licence file present: False) -> `data_registry/D04_autism_prediction_licence_citation.txt`

## Folder tree (first 3 levels)

| dir | depth | files | subdirs |
|---|---|---|---|
| `.` | 0 | 2 | 0 |

## `test.csv`

- Shape: [200, 21]
- ID columns: ['ID'] (unique: {'ID': True})
- Label column(s): NONE
- Class counts: NO label column in this file (unlabelled).
- Duplicate rows: full=0, excluding ID=0
- Fully empty columns: none
- '?' placeholder values: {'ethnicity': 54, 'relation': 6}

| column | dtype | missing | unique values (first 20) |
|---|---|---|---|
| `ID` | int64 | 0 | (numeric/continuous) |
| `A1_Score` | int64 | 0 | ['0', '1'] |
| `A2_Score` | int64 | 0 | ['0', '1'] |
| `A3_Score` | int64 | 0 | ['0', '1'] |
| `A4_Score` | int64 | 0 | ['0', '1'] |
| `A5_Score` | int64 | 0 | ['0', '1'] |
| `A6_Score` | int64 | 0 | ['0', '1'] |
| `A7_Score` | int64 | 0 | ['0', '1'] |
| `A8_Score` | int64 | 0 | ['0', '1'] |
| `A9_Score` | int64 | 0 | ['0', '1'] |
| `A10_Score` | int64 | 0 | ['0', '1'] |
| `age` | float64 | 0 | (numeric/continuous) |
| `gender` | str | 0 | ['f', 'm'] |
| `ethnicity` | str | 0 | ['?', 'Asian', 'Black', 'Hispanic', 'Latino', 'Middle Eastern ', 'Others', 'Pasifika', 'South Asian', 'Turkish', 'White-European'] |
| `jaundice` | str | 0 | ['no', 'yes'] |
| `austim` | str | 0 | ['no', 'yes'] |
| `contry_of_res` | str | 0 | (numeric/continuous) |
| `used_app_before` | str | 0 | ['no', 'yes'] |
| `result` | float64 | 0 | (numeric/continuous) |
| `age_desc` | str | 0 | ['18 and more'] |
| `relation` | str | 0 | ['?', 'Health care professional', 'Others', 'Parent', 'Relative', 'Self'] |

## `train.csv`

- Shape: [800, 22]
- ID columns: ['ID'] (unique: {'ID': True})
- Label column(s): ['Class/ASD']
- Class counts: {'0': 639, '1': 161}
- Duplicate rows: full=0, excluding ID=0
- Fully empty columns: none
- '?' placeholder values: {'ethnicity': 203, 'relation': 40}

| column | dtype | missing | unique values (first 20) |
|---|---|---|---|
| `ID` | int64 | 0 | (numeric/continuous) |
| `A1_Score` | int64 | 0 | ['0', '1'] |
| `A2_Score` | int64 | 0 | ['0', '1'] |
| `A3_Score` | int64 | 0 | ['0', '1'] |
| `A4_Score` | int64 | 0 | ['0', '1'] |
| `A5_Score` | int64 | 0 | ['0', '1'] |
| `A6_Score` | int64 | 0 | ['0', '1'] |
| `A7_Score` | int64 | 0 | ['0', '1'] |
| `A8_Score` | int64 | 0 | ['0', '1'] |
| `A9_Score` | int64 | 0 | ['0', '1'] |
| `A10_Score` | int64 | 0 | ['0', '1'] |
| `age` | float64 | 0 | (numeric/continuous) |
| `gender` | str | 0 | ['f', 'm'] |
| `ethnicity` | str | 0 | ['?', 'Asian', 'Black', 'Hispanic', 'Latino', 'Middle Eastern ', 'Others', 'Pasifika', 'South Asian', 'Turkish', 'White-European', 'others'] |
| `jaundice` | str | 0 | ['no', 'yes'] |
| `austim` | str | 0 | ['no', 'yes'] |
| `contry_of_res` | str | 0 | (numeric/continuous) |
| `used_app_before` | str | 0 | ['no', 'yes'] |
| `result` | float64 | 0 | (numeric/continuous) |
| `age_desc` | str | 0 | ['18 and more'] |
| `relation` | str | 0 | ['?', 'Health care professional', 'Others', 'Parent', 'Relative', 'Self'] |
| `Class/ASD` | int64 | 0 | ['0', '1'] |

**Leakage check:** No exact single-feature rule over the whole file; best is 'ASD if sum(A1_Score..A10_Score) >= 10' accuracy 0.8625 (majority baseline 0.79875)

| feature | best rule | accuracy | [[TN,FP],[FN,TP]] |
|---|---|---|---|
| sum(A1_Score..A10_Score) | ASD if sum(A1_Score..A10_Score) >= 10 | 0.8625 | [[613, 26], [84, 77]] |
| age | ASD if age >= 80.5441 | 0.8 | [[637, 2], [158, 3]] |
| result | ASD if result >= 15.0786 | 0.8 | [[631, 8], [152, 9]] |

- item_sum_equals_result: {'rows_equal': 0, 'n': 800, 'max_abs_diff': 15.49142133}

- Majority-class baseline accuracy: 0.79875
