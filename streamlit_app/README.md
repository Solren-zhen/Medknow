# MedKnow Streamlit demo

Free, forever, on [Streamlit Community Cloud](https://share.streamlit.io) — no
Hugging Face account needed.

## Deploy (2 minutes)

1. Push this repository to GitHub (done).
2. Go to **https://share.streamlit.io** → **Create app** → connect this repo
   (file: `streamlit_app/app.py`, main module: `app`).
3. Done. The app auto-downloads the seed_42 weights (~43 MB) from the GitHub
   Release on first run, then caches them.

## Local run

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

> The interactive HF-Space version (`hf_space/`) and the one-click Colab demo
> (`notebooks/`) are the other two demo entry points.
