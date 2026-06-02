# Contributing

Thanks for your interest in improving this project. The repo started as a
learning exercise in 2017 and is now maintained as a small, modern reference
for predicting equity-index direction with a feed-forward neural network.

## Quick start

```bash
git clone https://github.com/drabekj/Stock-Marke-Prediction.git
cd Stock-Marke-Prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python download_data.py   # refresh local CSVs from Yahoo Finance
python train.py           # train and evaluate the model
```

## What kind of contributions are welcome

- Bug fixes in `train.py` or `download_data.py`.
- Replacing or extending the feature set (additional indices, macro data).
- Notebook improvements that keep it runnable on modern TensorFlow.
- Documentation, typo fixes, clearer comments.

## Style and process

- Run `ruff check .` before opening a PR.
- Keep changes small and focused — one idea per PR.
- Open an issue first if the change is large or alters the model interface.

## Reporting issues

Please include the Python version, OS, and the full traceback. Stock data
quality changes upstream (Yahoo occasionally rewrites history), so attach the
date you ran `download_data.py` if the bug involves training results.
