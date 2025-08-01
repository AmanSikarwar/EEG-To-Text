# EEG-To-Text and Sentiment Analysis

This project explores the translation of Electroencephalography (EEG) signals into natural language text and the classification of sentiment directly from EEG signals. It utilizes various deep learning models, primarily sequence-to-sequence architectures and classifiers, trained and evaluated on the ZuCo dataset.

## Features

* **EEG-to-Text Decoding:** Translates brain activity recorded via EEG into corresponding text using models like BrainTranslator and T5Translator.
* **EEG-based Sentiment Analysis:** Classifies sentiment (e.g., positive, negative, neutral) directly from EEG signals using baseline models (MLP, LSTM) and fine-tuned transformers.
* **Text-based Sentiment Classification:** Includes scripts for training standard text classifiers (BERT, BART, RoBERTa) on datasets like the Stanford Sentiment Treebank (SST) for comparison or zero-shot approaches.
* **Zero-Shot Sentiment Discovery:** Explores predicting sentiment from EEG by first decoding EEG to text and then classifying the generated text using a pre-trained text sentiment classifier.

## Models Implemented

* **Decoding Models:**
  * `BrainTranslator` (based on BART)
  * `BrainTranslatorNaive` (simplified BART-based)
  * `T5Translator` (based on T5)
* **Sentiment Models (EEG-based):**
  * `BaselineMLPSentence`
  * `BaselineLSTM`
  * `NaiveFineTunePretrainedBert` (EEG input adapted for BERT)
* **Sentiment Models (Text-based):**
  * Fine-tuned `BertForSequenceClassification`
  * Fine-tuned `BartForSequenceClassification`
  * Fine-tuned `RobertaForSequenceClassification`
* **Zero-Shot Pipeline:**
  * `ZeroShotSentimentDiscovery` (combines a decoder and a text classifier)

## Datasets

* **ZuCo Dataset:** The primary dataset containing EEG recordings synchronized with reading tasks (Task 1-SR, Task 2-NR, Task 3-TSR, Task 2-NR-2.0).
* **Stanford Sentiment Treebank (SST):** Used for training text-based sentiment classifiers. A filtered version is generated to avoid overlap with ZuCo sentences.

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/AmanSikarwar/EEG-To-Text
    cd EEG-To-Text
    ```

2. **Install Dependencies:** Ensure you have Python 3 and PyTorch installed. Install other required packages (a `requirements.txt` file might be needed):

    ```bash

    pip install torch transformers numpy tqdm evaluate nltk rouge-score scikit-learn yagmail h5py fuzzy_match
    ```

3. **Download Datasets:** Place the raw ZuCo `.mat` files in the appropriate `./dataset/ZuCo/` subdirectories (e.g., `./dataset/ZuCo/task1-SR/Matlab_files/`). Download the Stanford Sentiment Treebank dataset if needed and place it in `./dataset/stanfordsentiment/`.

## Data Preparation

Run the preparation script to convert ZuCo `.mat` files into `.pickle` format and generate necessary sentiment label files:

```bash
bash scripts/prepare_dataset.sh
```

This script utilizes utilities in the `./util/` directory (`construct_dataset_mat_to_pickle_v1.py`, `construct_dataset_mat_to_pickle_v2.py`, `get_sentiment_labels.py`, `get_SST_ternary_dataset.py`). Processed data will be stored in `./dataset/ZuCo/.../pickle/` and `./dataset/stanfordsentiment/`.

## Training

Training is performed using Python scripts. Configuration is often managed via command-line arguments and saved `.json` files in `./config/`. Checkpoints are saved in `./checkpoints/`.

* **EEG-to-Text Decoding:**
  * Use `train_decoding2.py` (or potentially `train_decoding.py`).
  * Example script: `run_and_notify2.sh` shows how to run `train_decoding2.py` with specific parameters for a 2-step training process.
* **EEG Sentiment Baseline:**
  * Use `train_sentiment_baseline.py`.
  * Example script: `scripts/train_eeg_sentiment_baseline.sh`
* **Text-based Sentiment Classifier:**
  * Use `train_sentiment_textbased.py`.
  * Can be trained on ZuCo text or SST.

Refer to the scripts in the `scripts/` directory and `run_and_notify2.sh` for detailed examples of training commands and parameters.

## Evaluation

Evaluation scripts measure the performance using various metrics (BLEU, ROUGE, WER, CER for decoding; Accuracy, F1 for sentiment). Results are typically saved in `./results/` and `./score_results/`.

* **EEG-to-Text Decoding:**
  * Use `eval_decoding2.py` (or potentially `eval_decoding.py`).
  * Example usage is shown within `run_and_notify2.sh`.
* **Sentiment Analysis:**
  * Use `eval_sentiment.py`.
  * Example script for zero-shot evaluation: `scripts/eval_sentiment_zeroshot_pipeline.sh`

## Monitoring

The `run_and_notify2.sh` script provides an example of how to run a full training and evaluation pipeline and send email notifications upon completion or failure using `yagmail`. Configure sender/recipient emails within the script. Logs are stored in `./run_logs/`.

## Directory Structure

```text
.
├── checkpoints/       # Saved model weights (.pt files)
│   ├── decoding/
│   └── eeg_sentiment/
│   └── text_sentiment_classifier/
├── config/            # Configuration files (.json) for experiments
│   ├── decoding/
│   └── eeg_sentiment/
│   └── text_sentiment_classifier/
├── data.py            # PyTorch Dataset and DataLoader classes (ZuCo_dataset, SST_tenary_dataset)
├── dataset/           # Raw and processed datasets
│   ├── ZuCo/
│   └── stanfordsentiment/
├── eval_decoding.py   # Evaluation script for EEG-to-Text (older version?)
├── eval_decoding2.py  # Main evaluation script for EEG-to-Text
├── eval_sentiment.py  # Evaluation script for sentiment classification tasks
├── model_decoding.py  # Decoding model definitions (BrainTranslator, T5Translator, etc.)
├── model_sentiment.py # Sentiment model definitions (Baselines, ZeroShot, etc.)
├── results/           # Detailed output files from evaluation runs (.txt)
├── run_logs/          # Logs generated by run_and_notify scripts
├── run_and_notify.sh  # Example script for running experiments with notifications (older?)
├── run_and_notify2.sh # Example script for running experiments with notifications
├── score_results/     # Summarized score files from evaluation runs (.txt)
├── scripts/           # Helper shell scripts for training/evaluation/preparation
├── train_decoding.py  # Training script for EEG-to-Text (older version?)
├── train_decoding2.py # Main training script for EEG-to-Text
├── train_sentiment_baseline.py # Training script for EEG sentiment baselines
├── train_sentiment_textbased.py # Training script for text-based sentiment classifiers
└── util/              # Utility scripts for data processing, etc.
```
