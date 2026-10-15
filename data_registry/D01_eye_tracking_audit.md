# D01_eye_tracking audit

- Folder: `Eye-Tracking Dataset`
- Detected data type: **eye_tracking_timeseries_csv**
- Files: 27 | size: 667.31 MB | max depth: 2
- Files by extension: {'.csv': 26, '.txt': 1}
- Archives (zip/rar): none
- Licence/citation files: ['Citation.txt'] (licence file present: False) -> `data_registry/D01_eye_tracking_licence_citation.txt`

## Folder tree (first 3 levels)

| dir | depth | files | subdirs |
|---|---|---|---|
| `.` | 0 | 2 | 1 |
| `Eye-tracking Output` | 1 | 25 | 0 |

## Eye-tracking

- Gaze CSV files: 25 ; total rows: 2252943
- Column-set variants: 37 cols -> files ['1.csv', '6.csv', '10.csv', '16.csv']; 51 cols -> files ['2.csv', '4.csv', '12.csv', '14.csv', '15.csv', '19.csv', '22.csv', '24.csv']; 52 cols -> files ['3.csv', '11.csv', '21.csv']; 51 cols -> files ['5.csv', '9.csv', '18.csv']; 58 cols -> files ['7.csv']; 39 cols -> files ['8.csv']; 38 cols -> files ['13.csv']; 52 cols -> files ['17.csv', '20.csv', '23.csv']; 50 cols -> files ['25.csv']
- Metadata: `Metadata_Participants.csv` columns ['ParticipantID', 'Gender', 'Age', 'Class', 'CARS Score']; 59 participants; classes {'TD': 30, 'ASD': 29}; gender {'M': 38, 'F': 21}; age {'min': 2.7, 'median': 8.1, 'max': 12.9}; missing {'ParticipantID': 0, 'Gender': 0, 'Age': 0, 'Class': 0, 'CARS Score': 30}
- Unique Participant values in gaze: 59 (numeric IDs: 57)
- Non-numeric Participant rows: {'Unidentified(Neg)': 878362, 'Unidentified(Pos)': 23424}
- Participants with metadata AND gaze: 57 {'ASD': 27, 'TD': 30}
- Metadata but no gaze rows: [12, 16]
- Gaze IDs without metadata: none
- Gaze rows by class: {'TD': 857767, 'ASD': 493390, 'unidentified': 901786}
- Byte-identical gaze files (md5): none
- Distinct row keys (RecordingTime [ms], Time of Day, Participant, Trial): 1923909 of 2252943 rows
- Sampling: median dt 16.700000000186265 ms (~59.88 Hz), p05 16.68500000005588, p95 44.0

| file | rows | cols | trials | unidentified rows | rows shared with other files | median dt ms | participants |
|---|---|---|---|---|---|---|---|
| 1.csv | 34630 | 37 | 1 | 4765 | 34630 | 20.005000000121072 | 15, 17, 18, 19, 20, 21, 23, 27, 28, 42, 46, 49, 50, 52, 53, 55, Unidentified(Neg), Unidentified(Pos) |
| 2.csv | 16687 | 51 | 1 | 3938 | 0 | 19.887999999977183 | 11, 30, 31, 33, 37, 39, Unidentified(Neg) |
| 3.csv | 124658 | 52 | 1 | 74782 | 0 | 16.693000000028405 | 25, 26, 29, 30, 31, 36, 49, 52, 54, 56, 57, 58, 59, Unidentified(Neg) |
| 4.csv | 23136 | 51 | 1 | 3268 | 17651 | 19.910999999847263 | 15, 17, 18, 19, 20, 21, 42, 49, 50, 52, Unidentified(Neg), Unidentified(Pos) |
| 5.csv | 17769 | 51 | 1 | 3800 | 261 | 19.932000000029802 | 11, 23, 27, 30, 31, 33, 37, 39, Unidentified(Neg) |
| 6.csv | 62648 | 37 | 18 | 5378 | 0 | 19.87800000049174 | 13, 14, 15, 17, 18, 19, 20, 21, 23, 27, 28, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, Unidentified(Neg), Unidentified(Pos) |
| 7.csv | 61583 | 58 | 1 | 29319 | 504 | 16.699999999953434 | 24, 25, 26, 29, 30, 54, 56, 58, 59, Unidentified(Neg) |
| 8.csv | 114285 | 39 | 1 | 67352 | 114285 | 16.693000000028405 | 25, 26, 29, 30, 31, 36, 49, 52, 54, 56, 57, 58, 59, Unidentified(Neg) |
| 9.csv | 74479 | 51 | 1 | 24146 | 245 | 19.88099999993574 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, Unidentified(Neg) |
| 10.csv | 51829 | 37 | 1 | 5065 | 51828 | 19.895000000018626 | 13, 14, 15, 17, 18, 19, 20, 21, 23, 27, 28, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, Unidentified(Neg), Unidentified(Pos) |
| 11.csv | 114285 | 52 | 1 | 67352 | 114285 | 16.693000000028405 | 25, 26, 29, 30, 31, 36, 49, 52, 54, 56, 57, 58, 59, Unidentified(Neg) |
| 12.csv | 73440 | 51 | 1 | 22087 | 0 | 19.87999999988824 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 53, Unidentified(Neg) |
| 13.csv | 21947 | 38 | 14 | 0 | 0 | 19.878000000026077 | 11, 41, 49 |
| 14.csv | 28374 | 51 | 14 | 0 | 22171 | 19.88399999961257 | 17, 18, 19, 20, 21 |
| 15.csv | 29699 | 51 | 14 | 13938 | 0 | 19.879000000015367 | 37, 50, Unidentified(Neg) |
| 16.csv | 110967 | 37 | 15 | 21238 | 34630 | 19.885000000009313 | 15, 17, 18, 19, 20, 21, 23, 27, 28, 42, 46, 49, 50, 52, 53, 55, Unidentified(Neg), Unidentified(Pos) |
| 17.csv | 396298 | 52 | 14 | 235790 | 0 | 16.692999999970198 | 22, 24, 29, 30, 31, 36, 49, 52, 54, 56, 57, 58, 59, Unidentified(Neg) |
| 18.csv | 70797 | 51 | 18 | 21855 | 294 | 19.877999999793246 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, Unidentified(Neg) |
| 19.csv | 71234 | 51 | 18 | 21669 | 0 | 19.87699999997858 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 53, Unidentified(Neg) |
| 20.csv | 123702 | 52 | 17 | 73664 | 0 | 16.693000000028405 | 22, 24, 29, 30, 31, 36, 49, 52, 54, 56, 57, 58, 59, Unidentified(Neg) |
| 21.csv | 284931 | 52 | 34 | 80283 | 109278 | 19.867000000027474 | 13, 14, 17, 18, 19, 20, 21, 23, 25, 26, 27, 28, 29, 30, 31, 36, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, Unidentified(Neg) |
| 22.csv | 56929 | 51 | 17 | 3504 | 37039 | 19.87900000065565 | 13, 14, 15, 17, 18, 19, 20, 21, 28, 32, 35, 36, 38, 40, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, Unidentified(Neg), Unidentified(Pos) |
| 23.csv | 130909 | 52 | 1 | 77941 | 0 | 16.693000000086613 | 22, 24, 29, 30, 31, 36, 49, 52, 54, 56, 57, 58, 59, Unidentified(Neg) |
| 24.csv | 38663 | 51 | 1 | 3358 | 31617 | 19.895000000018626 | 13, 14, 15, 17, 18, 19, 20, 21, 28, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, Unidentified(Neg), Unidentified(Pos) |
| 25.csv | 119064 | 50 | 1 | 37294 | 51828 | 16.699999999720603 | 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, Unidentified(Neg), Unidentified(Pos) |
