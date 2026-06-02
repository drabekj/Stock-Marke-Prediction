# Stock Market Prediction

> Feed-forward neural network that predicts the daily direction of the S&P 500
> closing index from the same-day closes of earlier-closing world indices.

[![CI](https://github.com/drabekj/Stock-Marke-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/drabekj/Stock-Marke-Prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://www.tensorflow.org/)

The project started in 2017 as a hands-on study of how time-zone–staggered
market closes can leak short-term information about the S&P 500. It has been
modernised for Python 3.11 + TensorFlow 2 and a `yfinance`-based downloader
so it still works after Yahoo and Quandl deprecated the original APIs.

> Note on the repo name: the original spelling (`Stock-Marke-Prediction`) is
> kept to preserve stars and external links. The project itself is, of
> course, about Stock _Market_ Prediction.

---

## The idea

Several major equity indices close before the S&P 500 does. Their same-day
closes are therefore knowable at the moment we want to predict the S&P 500's
own close.

| Index                                 | Region    | Close (EST) |
|---------------------------------------|-----------|-------------|
| All Ordinaries (`^AORD`)              | Australia | 01:00       |
| Nikkei 225 (`^N225`)                  | Japan     | 02:00       |
| Hang Seng (`^HSI`)                    | Hong Kong | 04:00       |
| DAX (`^GDAXI`)                        | Germany   | 11:30       |
| NYSE Composite (`^NYA`)               | US        | 16:00       |
| Dow Jones Industrial (`^DJI`)         | US        | 16:00       |
| **S&P 500 (`^GSPC`)** — target        | US        | 16:00       |

The model uses the daily returns of the non-target indices over a short
lookback window to classify whether the S&P 500's return on the same day will
be positive or negative.

---

## Quick start

```bash
git clone https://github.com/drabekj/Stock-Marke-Prediction.git
cd Stock-Marke-Prediction

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Optional) refresh the local CSVs in data/ from Yahoo Finance.
python download_data.py

# Train + evaluate.
python train.py
```

Example output:

```
Test accuracy: 0.5512  (majority-class baseline: 0.5293)
```

The original walkthrough is in [`Globe+Markets.ipynb`](Globe+Markets.ipynb).
`train.py` is a standalone TensorFlow 2 / Keras re-implementation that runs
end-to-end without the deprecated `tf.placeholder` / `tf.contrib` APIs.

---

## Repository layout

```
.
├── train.py                # Standalone TF2 training script (modern entry point)
├── download_data.py        # Refresh data/ from Yahoo Finance via yfinance
├── data/                   # Historical CSVs (2012–2017 snapshot included)
├── Globe+Markets.ipynb     # Original 2017 walkthrough notebook
├── requirements.txt        # Runtime deps (TF 2.16+, pandas 2.2+, ...)
├── requirements-dev.txt    # + jupyter, ruff, pytest
└── .github/workflows/ci.yml
```

---

## Roadmap

- [ ] Replace the dense network with a tiny LSTM for direct comparison.
- [ ] Walk-forward validation in place of the single train/test split.
- [ ] Add per-feature ablation to quantify how much each foreign index helps.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) if you'd like to help.

---

## License

[MIT](LICENSE).
