# Data

**No imaging data is distributed in this repository.** Chest radiographs are
medical data governed by each source's license and terms of use. This folder
contains only provenance notes; the patient-level split definitions are kept in
`data/split_patient/` (split CSVs + `split_report.json`) and are safe to
distribute.

## Datasets

| Split | Dataset | Source | Access | Notes |
|---|---|---|---|---|
| Internal | Chest X-Ray Images (Kermany et al., 2018) | [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) | Kaggle download | 5,856 frontal chest radiographs, pediatric-centric, binary pneumonia/normal |
| External #1 | RSNA Pneumonia Detection Challenge (Shih et al., 2019) | [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) | Competition terms of use | 26,684 images; classes Lung Opacity / No Lung Opacity / Normal |
| External #2 | NIH ChestXray-14 (Wang et al., 2017) | [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) | Official download | Two-class Pneumonia / No Finding subset, 9,103 test images |
| Optional | VinDr-CXR | [VinBigData](https://www.kaggle.com/c/vindr-cxr) | Kaggle | Present locally; **not used in the manuscript** |

## Patient-level split (internal)

The internal dataset ships with an image-level split, which permits the same
patient to appear in both training and test partitions (leakage). This
repository re-splits by patient identifier (filename prefix) at 70/15/15 with
seed 42:

- train: 4,076 images (2,222 patients)
- val: 884 images (476 patients)
- test: 896 images (476 patients), pneumonia prevalence 75.6%

Regenerate with:

```bash
python scripts/medknow_split_patient.py --seed 42
```

## Usage restrictions

- Do not upload any image, DICOM, or patient information to this repository.
- Public datasets retain their original licenses; cite the sources (see
  `paper/_src/refs.bib`) when using them.
- This project is research/education only and is not a medical device.
