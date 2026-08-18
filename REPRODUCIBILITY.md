# Reproducibility manifest

The reference environment is Python 3.11 with PyTorch 2.5.1 and torchvision
0.20.1. These versions are aligned across `environment.yml`, the requirements
files, and the GitHub Actions CPU test job. The remaining scientific Python
dependencies follow the minimum versions in `pyproject.toml`; CI runs lint and
the full test suite on every push and pull request.

For a new experiment, record all of the following alongside the result JSON:

- repository commit (`git rev-parse HEAD`);
- Python, PyTorch, torchvision, NumPy, SciPy, and scikit-learn versions;
- dataset manifest or split report and its SHA-256 hash;
- checkpoint SHA-256 and the checkpoint's embedded `metadata` object;
- config file, random seed, device, and whether MC Dropout/temperature scaling
  was enabled;
- evaluation unit (image or patient/group) and bootstrap settings.

Training checkpoints created by `medknow.training.trainer` now embed model
architecture, class order, input size, normalization, dropout, seed, Python,
PyTorch, and protocol configuration metadata. No dataset images or patient
identifiers are committed to this repository.
