# MedKnow data card

## Data sources

MedKnow does not redistribute medical images. The pipeline can evaluate local
copies of the Kermany Chest X-Ray Images dataset, the RSNA Pneumonia Detection
Challenge, and a two-class NIH ChestXray-14 subset. See `data/README.md` for
source links, licenses, and download requirements.

## Label definitions

- Kermany: `PNEUMONIA` versus `NORMAL` as supplied by the dataset.
- RSNA: `Lung Opacity` is treated as positive; `No Lung Opacity / Not Normal`
  and `Normal` are treated as negative for the binary endpoint.
- NIH: the repository's prepared two-class manifest defines the positive and
  negative folders; the source labels are NLP-derived and noisy.

## Splitting and evaluation units

The internal split utility extracts patient identifiers from filenames and
keeps each patient in exactly one partition. RSNA records are deduplicated by
patient ID. Evaluation utilities support cluster bootstrap through an explicit
`groups` array; users must provide verified patient identifiers for any new
cohort.

## Known biases and limitations

The sources differ in age, prevalence, devices, views, institutions, label
workflows, and disease spectrum. These differences are part of the domain-shift
benchmark and mean that pooled metrics should not be interpreted as clinical
validation. Missing metadata, label noise, and selection of publicly available
images can bias subgroup and calibration analyses.

## Governance

Only de-identified or legally authorised local data should be used. Do not
upload patient images to an untrusted deployment. This project is for research
and education only.
