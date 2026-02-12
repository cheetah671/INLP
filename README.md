# INLP — Lyrics & Audio-Aware Language Modeling

A research and engineering repository for training, evaluating and deploying a lyric-generation language model that integrates textual and audio-derived features. The project contains datasets (English and Hindi), model training code and notebooks, and inference utilities used to produce the final model and a minimal app for serving generation.

Table of Contents
- Project overview
- Repository structure
- Data description
- Models, training & notebooks
- Inference & demo
- Quickstart (environment & common commands)
- How to reproduce training and evaluation
- Notes on model files and tokenizer
- Contribution guidelines
- License & contact

**Project overview**

This repository supports experiments for generating song lyrics conditioned on text and (optionally) audio-derived features. It contains preprocessing and dataset artifacts, exploratory and training notebooks, training/evaluation scripts, and utilities used to produce a final inference model and a simple app for generating lyrics.

Key goals:
- Provide end-to-end code and data for fine-tuning a transformer-based language model on lyric data.
- Support multilingual (English + Hindi) datasets and provide separate dataset artifacts for easy reuse.
- Include audio feature CSVs used for experiments that combine acoustic features with lyrics.

Repository structure (high level)
- Root: model and tokenizer artifacts: [config.json](config.json), [generation_config.json](generation_config.json), [tokenizer_config.json](tokenizer_config.json), [vocab.json](vocab.json), [added_tokens.json](added_tokens.json)
- `code/` — Development notebooks and scripts used for dataset creation and experiments. See [code/Dataset/](code/Dataset/) and [code/Models/](code/Models/).
- `code/Dataset/final_data/` — Cleaned dataset artifacts (English and Hindi) used for fine-tuning.
- `final_model/` — Final inference scripts, minimal app, and notebooks used to evaluate and serve the trained model. Notable files: [final_model/app.py](final_model/app.py), [final_model/generate.py](final_model/generate.py), [final_model/lyrics_generator.py](final_model/lyrics_generator.py), [final_model/test.py](final_model/test.py)
- `Transformers_interim_progress/` — Working notebooks and intermediate experiments using the HuggingFace Transformers library.

Data description

Datasets are available under `code/Dataset/final_data/` (and mirrored in `Transformers_interim_progress/code/Dataset/final_data/`). They include:
- English: CSVs with audio-derived features (e.g. `combined_audio_features.csv`, `merged_audio_features_lyrics.csv`) and intermediate CSV splits like `audio_features_1-300.csv` etc.
- Hindi: `audio_features_hindi.csv`, `formatted_lyrics.txt`, `gpt2_finetune_data.jsonl`, and `hindi_songs_with_sentiment.csv`/`.json`.

These artifacts are intended for: tokenization, preparing text sequences for autoregressive training, and joining acoustic features to text records when running multimodal experiments.

Models, training & notebooks

- Notebooks used for experiments are in `code/Models/` and `Transformers_interim_progress/code/Models/` (example: `lyric-gpt_midpoint.ipynb`).
- Training scripts and helper utilities appear in `code/Models/` and `final_model/`.
- The repository stores transformer config and tokenizer artifacts at the repository root so the final model can be reloaded using the Hugging Face transformers API.

Inference & demo

The `final_model/` directory contains quick utilities to run inference and a minimal app.
- Start the demo app (if `app.py` is a Flask/FastAPI entrypoint):

```bash
# create a venv and activate it
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run the app (example)
python final_model/app.py
```

- To run the generation script directly (example):

```bash
python final_model/generate.py --prompt "Your lyric prompt here"
```

Note: Command-line arguments vary by script. Inspect the script headers or open the corresponding notebooks to see available options.

Quickstart — environment & common commands

1. Create and activate Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install common dependencies (suggested):

```bash
pip install transformers torch datasets pandas numpy scikit-learn flask jupyterlab
```

3. Run notebooks for preprocessing and training:

```bash
# open Jupyter Lab
jupyter lab
```

How to reproduce training and evaluation

1. Prepare your dataset: run the preprocessing notebook `code/Dataset/create_final_english_dataset.ipynb` or the corresponding Hindi preprocessing notebook to produce `gpt2_finetune_data.jsonl` or `merged_audio_features_lyrics.csv`.
2. Tokenize with the repository tokenizer artifacts (root-level tokenizer files) or reinitialize a new tokenizer in the notebook.
3. Use the training notebooks in `code/Models/` or a script (if available) to fine-tune a transformer (e.g., GPT-2 variant) on the prepared `.jsonl` file.
4. Save the trained model and push its config/tokenizer files into a model directory, then update `final_model/` to point to the saved checkpoint for inference.

Notes on model and tokenizer files

- The root contains tokenizer and model config artifacts so code in `final_model/` can load a local model without requiring a remote Hugging Face hub download.
- If you re-train or fine-tune, ensure you save both the model weights and tokenizer files to the `final_model/` model directory (or update the code to point to your checkpoint path).

Where to look next (important files)
- Data preprocessing: [code/Dataset/create_final_english_dataset.ipynb](code/Dataset/create_final_english_dataset.ipynb)
- English dataset artifacts: [code/Dataset/final_data/English/combined_audio_features.csv](code/Dataset/final_data/English/combined_audio_features.csv)
- Hindi dataset artifacts: [code/Dataset/final_data/Hindi/gpt2_finetune_data.jsonl](code/Dataset/final_data/Hindi/gpt2_finetune_data.jsonl)
- Final inference: [final_model/generate.py](final_model/generate.py) and [final_model/app.py](final_model/app.py)

Contributing

- Please open an issue for feature requests or bug reports.
- If you send a pull request, include a clear description of the change and add/modify notebooks or small scripts demonstrating behavior where relevant.

License & contact

This repository does not include a license file by default. If you want to make the code public, add a `LICENSE` file (MIT, Apache-2.0, or another license you prefer).

For questions, reach out to the maintainer listed in the repository or open an issue.

— End of README —
