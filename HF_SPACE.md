# Deploy the demo to Hugging Face Spaces (free, ~5 minutes)

The live demo runs entirely in the browser — no install, no GPU, no data download.
This is the single biggest lever for getting the repo noticed.

> ✅ **Ready-made package:** everything below is already prepared in
> [`hf_space/`](hf_space/) (self-contained `app.py` + `requirements.txt` +
> model-card `README.md`). You only need to upload the model weights.

## Steps

1. **Create a Space** at <https://huggingface.co/new-space>
   - SDK: **Gradio**, hardware: **CPU basic** (free) is enough
   - Name it e.g. `medknow-pneumonia-xray`

2. **Push the files from `hf_space/`** to the Space root (git push or web UI upload):
   - `app.py`
   - `requirements.txt`
   - `README.md` (the model card renders on the Space page)
   - `examples/` — copy the `examples/*.jpeg` images from the repo root

3. **Upload the model weights** (required once, ~45 MB):
   - Copy `outputs/pneumonia_model.pth` (seed_42 deployment checkpoint) from
     `pneumonia_classifier\` and upload it as `model.pth` at the Space root.
     (`app.py` detects `model.pth` automatically; files > 10 MB are handled via
     git LFS or the web UI "Upload files" button.)
   - Optional but recommended: also upload
     `outputs/temperature.txt` (T≈1.67, matches `pneumonia_model.pth`) as
     `temperature.txt` so temperature scaling is applied.

4. **Link the Space from your GitHub README** — top badge:

   ```markdown
   [![Live Demo](https://img.shields.io/badge/🖥️-Live%20Demo-FF4B4B)](https://huggingface.co/spaces/ojdanajakir848-a11y/medknow-pneumonia-xray)
   ```

## Tips

- Free CPU Spaces are slow with 30 MC samples. If the Space times out, lower the
  default in `hf_space/app.py` (`DEFAULT_MC_SAMPLES = 30`) or set
  `MODEL_PATH` / `TEMPERATURE_PATH` env vars (defaults already point to the
  Space root).
- Space traffic is separate from GitHub traffic: a popular Space *drives* stars
  to the repo, so promote the Space link, not just the GitHub link.
- Keep the Space public; private Spaces do not show up in discovery.
