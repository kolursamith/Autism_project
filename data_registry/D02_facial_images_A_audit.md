# D02_facial_images_A audit

- Folder: `Autism Spectrum Disorder Detection using Facial Traits Dual Framework DenseRes with K-Fold Cross Validation (DFDK) for Enhanced Classification Accuracy`
- Detected data type: **images**
- Files: 2950 | size: 117.35 MB | max depth: 5
- Files by extension: {'.jpg': 2940, '.png': 10}
- Archives (zip/rar): none
- Licence/citation files: none (licence file present: False) -> `data_registry/D02_facial_images_A_licence_citation.txt`

## Folder tree (first 3 levels)

| dir | depth | files | subdirs |
|---|---|---|---|
| `.` | 0 | 10 | 1 |
| `AutismDataset` | 1 | 0 | 1 |
| `AutismDataset/AutismDataset` | 2 | 0 | 3 |
| `AutismDataset/AutismDataset/test` | 3 | 0 | 2 |
| `AutismDataset/AutismDataset/train` | 3 | 0 | 2 |
| `AutismDataset/AutismDataset/valid` | 3 | 0 | 2 |

## Images

- Samples (images inside class folders): 2940
- Image files outside class folders (not samples): ['CF.png', 'c1.png', 'cmdense2.png', 'f1.png', 'lossnew.png', 'reportdense.png', 'resnextacc.png', 'resnextloss.png', 'roc.png', 'roc_resnext.png']
- Class folder names per split (raw): {'train': ['Autistic', 'Non_Autistic'], 'valid': ['Autistic', 'Non_Autistic'], 'test': ['Autistic', 'Non-Autistic']}

| split | autistic | nonautistic | total |
|---|---|---|---|
| train | 1193 | 1193 | 2386 |
| valid | 127 | 127 | 254 |
| test | 150 | 150 | 300 |

- Width min/median/max: {'min': 114, 'median': 267.0, 'max': 2471} ; Height: {'min': 126, 'median': 309.0, 'max': 2872} ; distinct sizes: 2557
- Colour modes: {'RGB': 2940} ; unreadable files: 0
- **Exact duplicates (md5)**: 0 groups, 0 images, 0 redundant copies; within one split: 0, across splits: 0 {}; groups with label conflict: 0
- **Near duplicates (phash Hamming <= 4)**: 11 pairs (11 not byte-identical); within split 6, across splits 5 {'train|valid': 2, 'test|train': 3}; different class: 0; connected groups: 11
- Subject IDs: NO subject/participant ID exists: filenames are only <class>.<index>.jpg or <index>.jpg and the folder contains no CSV/metadata. The same child may appear in several images and cannot be grouped.
- Contact sheet: `data_registry/D02_facial_images_A_samples.png`

## Cross-dataset duplicates (D02 vs D03)

```json
{
 "datasets": [
  "D03_facial_images_B",
  "D02_facial_images_A"
 ],
 "exact_md5": {
  "D03_facial_images_B_images_with_exact_copy_in_D02_facial_images_A": 2541,
  "D03_facial_images_B_images_total": 2940,
  "D02_facial_images_A_unique_md5_total": 2940,
  "unique_md5_shared": 2541,
  "unique_md5_only_in_D03_facial_images_B": 399,
  "unique_md5_only_in_D02_facial_images_A": 399,
  "matched_D03_facial_images_B_images_by_split": {
   "none": 2541
  },
  "matched_images_with_same_class_label": 2541
 },
 "near_phash_cross_dataset_pairs(hamming<=4)": 2962,
 "D03_facial_images_B_images_with_near_or_exact_match_in_D02_facial_images_A(nearest hamming<=4)": 2940,
 "D03_facial_images_B_nearest_match_split_in_D02_facial_images_A": {
  "train": 2387,
  "valid": 253,
  "test": 300
 },
 "nearest_match_same_class": 2940,
 "nearest_distance_histogram": {
  "0": 2939,
  "2": 1
 },
 "D03_facial_images_B_non_byte_identical_images": {
  "n": 399,
  "nearest_distance_histogram": {
   "0": 398,
   "2": 1
  },
  "nearest_match_split": {
   "valid": 99,
   "test": 300
  },
  "same_pixel_size_as_nearest": 399,
  "examples(a, nearest_b, dist)": [
   [
    "consolidated/Autistic/0005.jpg",
    "AutismDataset/AutismDataset/valid/Autistic/01.jpg",
    0
   ],
   [
    "consolidated/Autistic/0006.jpg",
    "AutismDataset/AutismDataset/test/Autistic/Autistic.0.jpg",
    0
   ],
   [
    "consolidated/Autistic/0018.jpg",
    "AutismDataset/AutismDataset/test/Autistic/Autistic.1.jpg",
    0
   ],
   [
    "consolidated/Autistic/0050.jpg",
    "AutismDataset/AutismDataset/test/Autistic/Autistic.2.jpg",
    0
   ],
   [
    "consolidated/Autistic/0062.jpg",
    "AutismDataset/AutismDataset/test/Autistic/Autistic.3.jpg",
    0
   ],
   [
    "consolidated/Autistic/0074.jpg",
    "AutismDataset/AutismDataset/test/Autistic/Autistic.4.jpg",
    0
   ],
   [
    "consolidated/Autistic/0083.jpg",
    "AutismDataset/AutismDataset/test/Autistic/Autistic.5.jpg",
    0
   ],
   [
    "consolidated/Autistic/0110.jpg",
    "AutismDataset/AutismDataset/test/Autistic/Autistic.6.jpg",
    0
   ],
   [
    "consolidated/Autistic/0129.jpg",
    "AutismDataset/AutismDataset/test/Autistic/Autistic.7.jpg",
    0
   ],
   [
    "consolidated/Autistic/0147.jpg",
    "AutismDataset/AutismDataset/test/Autistic/Autistic.8.jpg",
    0
   ]
  ]
 }
}
```
