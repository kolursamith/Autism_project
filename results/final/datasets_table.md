# Datasets

| dataset | source_folder | modality | sample_unit | n_samples | n_positive | n_negative | group_unit | n_groups | n_groups_positive | n_groups_negative | role | exclusions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D01_eye_tracking | Eye-Tracking Dataset | eye-tracking gaze time series -> clips of 8 x 1-s gaze images | clip | 1241 | 437 | 804 | child | 57 | 27 | 30 | PRIMARY: 10 hybrid models | IDs 12, 16 have no gaze rows; rows without participant ID ('Unidentified', ~40%) dropped; cross-file duplicate rows removed (DD-02); CARS score never used (label proxy, DD-03) |
| D02_facial_images_A | DFDK facial images (AutismDataset sub-folder) | facial RGB images | image | 2940 | 1470 | 1470 | near-duplicate group | 2929 |  |  | SECONDARY: 10 hybrid models | original train/valid/test folders pooled (DD-05); pHash <= 4 pairs grouped (DD-07) |
| D04_autism_prediction | Autism_prediction dataset (train.csv) | screening questionnaire table | row | 800 | 161 | 639 | duplicate-feature group | 800 |  |  | tabular baselines | test.csv has no labels -> unused (DD-08) |
| D05_toddler_screening | Autism_dataset (Toddler Autism dataset July 2018.csv) | toddler screening questionnaire | row | 1054 | 728 | 326 | duplicate-feature group | 975 |  |  | tabular baselines (set_all = label-definitional upper bound; set_honest without A1-A10 / Qchat score) | label = (A1..A10 sum >= 4) (DD-09); Autism_Screening_Data_Combined.csv EXCLUDED: it embeds the toddler file with 728 labels changed |
| D06_early_autism | early_autism (asd.csv) | screening table | row | 1000 | 411 | 589 | duplicate-feature group | 1000 |  |  | tabular baselines | empty column 'Unnamed: 10' dropped (DD-10) |
| D03_facial_images_B | AutismDataset | facial RGB images | image | 2940 | 1470 | 1470 |  |  |  |  | EXCLUDED | copy of D02: 2,541 byte-identical + 399 re-saved images (DD-06) |

Sources: datasets_table.csv (column `source`).
