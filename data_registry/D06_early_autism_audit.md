# D06_early_autism audit

- Folder: `early_autism`
- Detected data type: **tabular**
- Files: 1 | size: 0.07 MB | max depth: 1
- Files by extension: {'.csv': 1}
- Archives (zip/rar): none
- Licence/citation files: none (licence file present: False) -> `data_registry/D06_early_autism_licence_citation.txt`

## Folder tree (first 3 levels)

| dir | depth | files | subdirs |
|---|---|---|---|
| `.` | 0 | 1 | 0 |

## `asd.csv`

- Shape: [1000, 11]
- ID columns: ['Child_ID'] (unique: {'Child_ID': True})
- Label column(s): ['Diagnosed_ASD']
- Class counts: {'No': 589, 'Yes': 411}
- Duplicate rows: full=0, excluding ID=0
- Fully empty columns: ['Unnamed: 10']
- '?' placeholder values: none

| column | dtype | missing | unique values (first 20) |
|---|---|---|---|
| `Child_ID` | str | 0 | (numeric/continuous) |
| `Age` | int64 | 0 | (numeric/continuous) |
| `Gender` | str | 0 | ['Female', 'Male'] |
| `Jaundice` | str | 0 | ['No', 'Yes'] |
| `Family_ASD_History` | str | 0 | ['No', 'Yes'] |
| `Language_Delay` | str | 0 | ['No', 'Yes'] |
| `Social_Interaction_Score` | int64 | 0 | ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9'] |
| `Communication_Score` | int64 | 0 | ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9'] |
| `Repetitive_Behavior_Score` | int64 | 0 | ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9'] |
| `Diagnosed_ASD` | str | 0 | ['No', 'Yes'] |
| `Unnamed: 10` | float64 | 1000 | [] |

**Leakage check:** No exact single-feature rule over the whole file; best is 'ASD if Age >= 35' accuracy 0.601 (majority baseline 0.589)

| feature | best rule | accuracy | [[TN,FP],[FN,TP]] |
|---|---|---|---|
| Age | ASD if Age >= 35 | 0.601 | [[553, 36], [363, 48]] |
| Communication_Score | ASD if Communication_Score <= 1 | 0.591 | [[538, 51], [358, 53]] |
| Repetitive_Behavior_Score | ASD if Repetitive_Behavior_Score >= 10 | 0.586 | [[557, 32], [382, 29]] |
| Social_Interaction_Score | ASD if Social_Interaction_Score <= 1 | 0.581 | [[541, 48], [371, 40]] |

- Majority-class baseline accuracy: 0.589
