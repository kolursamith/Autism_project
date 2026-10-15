# D05_toddler_screening audit

- Folder: `Autism_dataset`
- Detected data type: **tabular**
- Files: 3 | size: 0.32 MB | max depth: 1
- Files by extension: {'.csv': 2, '.docx': 1}
- Archives (zip/rar): none
- Licence/citation files: ['Toddler data description.docx'] (licence file present: False) -> `data_registry/D05_toddler_screening_licence_citation.txt`

## Folder tree (first 3 levels)

| dir | depth | files | subdirs |
|---|---|---|---|
| `.` | 0 | 3 | 0 |

## `Autism_Screening_Data_Combined.csv`

- Shape: [6075, 15]
- ID columns: none (unique: {})
- Label column(s): ['Class']
- Class counts: {'NO': 4271, 'YES': 1804}
- Duplicate rows: full=847, excluding ID=847
- Fully empty columns: none
- '?' placeholder values: none

| column | dtype | missing | unique values (first 20) |
|---|---|---|---|
| `A1` | int64 | 0 | ['0', '1'] |
| `A2` | int64 | 0 | ['0', '1'] |
| `A3` | int64 | 0 | ['0', '1'] |
| `A4` | int64 | 0 | ['0', '1'] |
| `A5` | int64 | 0 | ['0', '1'] |
| `A6` | int64 | 0 | ['0', '1'] |
| `A7` | int64 | 0 | ['0', '1'] |
| `A8` | int64 | 0 | ['0', '1'] |
| `A9` | int64 | 0 | ['0', '1'] |
| `A10` | int64 | 0 | ['0', '1'] |
| `Age` | int64 | 0 | (numeric/continuous) |
| `Sex` | str | 0 | ['f', 'm'] |
| `Jauundice` | str | 0 | ['no', 'yes'] |
| `Family_ASD` | str | 0 | ['no', 'yes'] |
| `Class` | str | 0 | ['NO', 'YES'] |

**Leakage check:** No exact single-feature rule over the whole file; best is 'ASD if sum(A1..A10) >= 7' accuracy 0.933827 (majority baseline 0.703045); PARTIAL LEAKAGE: within age bands the item-sum rule is exact for 5021 of 6075 rows

| feature | best rule | accuracy | [[TN,FP],[FN,TP]] |
|---|---|---|---|
| sum(A1..A10) | ASD if sum(A1..A10) >= 7 | 0.933827 | [[3869, 402], [0, 1804]] |
| Age | ASD if Age >= 80 | 0.70321 | [[4271, 0], [1803, 1]] |

- Majority-class baseline accuracy: 0.703045
- Age 0-3: n=1054 (ASD=0) best 'ASD if item_sum <= 0' accuracy 0.948767 CM [[1000, 54], [0, 0]]
- Age 4-11: n=1105 (ASD=529) best 'ASD if item_sum >= 7' accuracy 1.0 CM [[576, 0], [0, 529]]
- Age 12-17: n=711 (ASD=358) best 'ASD if item_sum >= 7' accuracy 1.0 CM [[353, 0], [0, 358]]
- Age 18-200: n=3205 (ASD=917) best 'ASD if item_sum >= 7' accuracy 1.0 CM [[2288, 0], [0, 917]]

## `Toddler Autism dataset July 2018.csv`

- Shape: [1054, 19]
- ID columns: ['Case_No'] (unique: {'Case_No': True})
- Label column(s): ['Class/ASD Traits ']
- Class counts: {'Yes': 728, 'No': 326}
- Duplicate rows: full=0, excluding ID=79
- Fully empty columns: none
- '?' placeholder values: none

| column | dtype | missing | unique values (first 20) |
|---|---|---|---|
| `Case_No` | int64 | 0 | (numeric/continuous) |
| `A1` | int64 | 0 | ['0', '1'] |
| `A2` | int64 | 0 | ['0', '1'] |
| `A3` | int64 | 0 | ['0', '1'] |
| `A4` | int64 | 0 | ['0', '1'] |
| `A5` | int64 | 0 | ['0', '1'] |
| `A6` | int64 | 0 | ['0', '1'] |
| `A7` | int64 | 0 | ['0', '1'] |
| `A8` | int64 | 0 | ['0', '1'] |
| `A9` | int64 | 0 | ['0', '1'] |
| `A10` | int64 | 0 | ['0', '1'] |
| `Age_Mons` | int64 | 0 | (numeric/continuous) |
| `Qchat-10-Score` | int64 | 0 | ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9'] |
| `Sex` | str | 0 | ['f', 'm'] |
| `Ethnicity` | str | 0 | ['Hispanic', 'Latino', 'Native Indian', 'Others', 'Pacifica', 'White European', 'asian', 'black', 'middle eastern', 'mixed', 'south asian'] |
| `Jaundice` | str | 0 | ['no', 'yes'] |
| `Family_mem_with_ASD` | str | 0 | ['no', 'yes'] |
| `Who completed the test` | str | 0 | ['Health Care Professional', 'Health care professional', 'Others', 'Self', 'family member'] |
| `Class/ASD Traits ` | str | 0 | ['No', 'Yes'] |

**Leakage check:** LEAKAGE: label reproduced exactly by 'ASD if sum(A1..A10) >= 4' (accuracy 1.0)

| feature | best rule | accuracy | [[TN,FP],[FN,TP]] |
|---|---|---|---|
| sum(A1..A10) | ASD if sum(A1..A10) >= 4 | 1.0 | [[326, 0], [0, 728]] |
| Qchat-10-Score | ASD if Qchat-10-Score >= 4 | 1.0 | [[326, 0], [0, 728]] |
| Age_Mons | ASD if Age_Mons >= 14 | 0.699241 | [[47, 279], [38, 690]] |

- item_sum_equals_Qchat-10-Score: {'rows_equal': 1054, 'n': 1054, 'max_abs_diff': 0.0}

- Majority-class baseline accuracy: 0.690702

## Cross-file finding: `Toddler Autism dataset July 2018.csv` is embedded in `Autism_Screening_Data_Combined.csv`

- All 1054 rows found, in order, at 0-based rows 5021..6074 of `Autism_Screening_Data_Combined.csv` (compared on ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Sex'])
- Labels there: {'no': 1054} vs original {'no': 326, 'yes': 728} -> **728 rows carry a different label**
