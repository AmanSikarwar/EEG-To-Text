import json
import os
import pickle
import time

import evaluate
import numpy as np
import torch
import torch.nn as nn
from evaluate import load
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from config import get_config
from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive, T5Translator

metric = evaluate.load("sacrebleu")
cer_metric = load("cer")
wer_metric = load("wer")


def remove_text_after_token(text, token="</s>"):
    token_index = text.find(token)
    if token_index != -1:
        return text[:token_index]
    return text


def eval_model(
    dataloaders,
    device,
    tokenizer,
    criterion,
    model,
    output_all_results_path="./results/temp.txt",
    score_results="./score_results/task.txt",
):
    start_time = time.time()
    model.eval()

    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    pred_tokens_list_previous = []
    pred_string_list_previous = []

    with open(output_all_results_path, "w") as f:
        for (
            input_embeddings,
            seq_len,
            input_masks,
            input_mask_invert,
            target_ids,
            target_mask,
            sentiment_labels,
        ) in tqdm(dataloaders["test"]):
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)

            target_tokens = tokenizer.convert_ids_to_tokens(
                target_ids_batch[0].tolist(), skip_special_tokens=True
            )
            target_string = tokenizer.decode(
                target_ids_batch[0], skip_special_tokens=True
            )

            f.write(f"target string: {target_string}\n")

            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)

            target_ids_batch_copy = target_ids_batch.clone().detach()
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            # Teacher-forcing evaluation (optional, for comparison)
            with torch.no_grad():
                seq2seqLMoutput = model(
                    input_embeddings_batch,
                    input_masks_batch,
                    input_mask_invert_batch,
                    target_ids_batch_copy,
                )
            logits_previous = seq2seqLMoutput.logits
            probs_previous = logits_previous[0].softmax(dim=1)
            values_previous, predictions_previous = probs_previous.topk(1)
            predictions_previous = torch.squeeze(predictions_previous)
            predicted_string_previous = remove_text_after_token(
                tokenizer.decode(predictions_previous)
                .split("</s></s>")[0]
                .replace("<s>", "")
            )
            f.write(f"predicted string with tf: {predicted_string_previous}\n")
            predictions_previous = predictions_previous.tolist()
            truncated_prediction_previous = []
            for t in predictions_previous:
                if t != tokenizer.eos_token_id:
                    truncated_prediction_previous.append(t)
                else:
                    break
            pred_tokens_previous = tokenizer.convert_ids_to_tokens(
                truncated_prediction_previous, skip_special_tokens=True
            )
            pred_tokens_list_previous.append(pred_tokens_previous)
            pred_string_list_previous.append(predicted_string_previous)

            # Generation evaluation (Beam Search)
            with torch.no_grad():
                if isinstance(model, nn.DataParallel):
                    generate_func = model.module.generate
                else:
                    generate_func = model.generate

                # Standard Hugging Face generate usually doesn't need target IDs for beam search.
                predictions_gen = generate_func(
                    input_embeddings_batch,
                    input_masks_batch,
                    input_mask_invert_batch,
                    target_ids_batch_copy,  # Pass the potentially required argument
                    max_length=56,
                    num_beams=5,
                    do_sample=False,
                    # repetition_penalty=1.0,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    early_stopping=True,
                )

            predicted_string = tokenizer.batch_decode(
                predictions_gen, skip_special_tokens=True
            )[0]

            # Use predictions_gen for token list if needed, encode the decoded string otherwise
            # Encoding the decoded string might be slightly different due to tokenizer nuances
            predictions_list = tokenizer.encode(
                predicted_string, add_special_tokens=False
            )

            f.write(f"predicted string: {predicted_string}\n")
            f.write("################################################\n\n\n")

            truncated_prediction = []
            # The generated predictions_gen often already handle EOS
            # If using the re-encoded list:
            for t in predictions_list:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)

            # Convert directly from generated IDs if they contain special tokens
            # pred_tokens = tokenizer.convert_ids_to_tokens(
            #     predictions_gen[0].tolist(), skip_special_tokens=True
            # )

            pred_tokens = tokenizer.convert_ids_to_tokens(
                truncated_prediction,
                skip_special_tokens=True,
            )

            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)

    weights_list = [
        (1.0,),
        (0.5, 0.5),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    corpus_bleu_scores = []
    corpus_bleu_scores_previous = []
    for i, weight in enumerate(weights_list):
        bleu_level = i + 1
        try:
            corpus_bleu_score = corpus_bleu(
                target_tokens_list, pred_tokens_list, weights=weight
            )
        except ZeroDivisionError:
            print(
                f"Warning: ZeroDivisionError calculating Corpus BLEU-{bleu_level}. Setting score to 0."
            )
            corpus_bleu_score = 0.0

        try:
            corpus_bleu_score_previous = corpus_bleu(
                target_tokens_list, pred_tokens_list_previous, weights=weight
            )
        except ZeroDivisionError:
            print(
                f"Warning: ZeroDivisionError calculating TF Corpus BLEU-{bleu_level}. Setting score to 0."
            )
            corpus_bleu_score_previous = 0.0

        corpus_bleu_scores.append(corpus_bleu_score)
        corpus_bleu_scores_previous.append(corpus_bleu_score_previous)
        print(f"corpus BLEU-{bleu_level} score: {corpus_bleu_score}")
        print(f"corpus BLEU-{bleu_level} score with tf: {corpus_bleu_score_previous}")

    reference_list = [[item] for item in target_string_list]

    sacre_blue = metric.compute(predictions=pred_string_list, references=reference_list)
    sacre_blue_previous = metric.compute(
        predictions=pred_string_list_previous, references=reference_list
    )
    print("\nsacreblue score: ", sacre_blue)
    print("sacreblue score with tf: ", sacre_blue_previous)

    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(
            pred_string_list, target_string_list, avg=True, ignore_empty=True
        )
    except ValueError:
        rouge_scores = {"Error": "Hypothesis is empty or invalid for ROUGE"}
    print("\nrouge_scores: ", rouge_scores)

    try:
        rouge_scores_previous = rouge.get_scores(
            pred_string_list_previous, target_string_list, avg=True, ignore_empty=True
        )
    except ValueError:
        rouge_scores_previous = {"Error": "TF Hypothesis is empty or invalid for ROUGE"}
    print("rouge_scores with tf:", rouge_scores_previous)

    try:
        wer_scores = wer_metric.compute(
            predictions=pred_string_list, references=target_string_list
        )
    except Exception as e:
        wer_scores = f"Error: {e}"
    print("\nWER score:", wer_scores)

    try:
        wer_scores_previous = wer_metric.compute(
            predictions=pred_string_list_previous, references=target_string_list
        )
    except Exception as e:
        wer_scores_previous = f"Error: {e}"
    print("WER score with tf:", wer_scores_previous)

    try:
        cer_scores = cer_metric.compute(
            predictions=pred_string_list, references=target_string_list
        )
    except Exception as e:
        cer_scores = f"Error: {e}"
    print("\nCER score:", cer_scores)

    try:
        cer_scores_previous = cer_metric.compute(
            predictions=pred_string_list_previous, references=target_string_list
        )
    except Exception as e:
        cer_scores_previous = f"Error: {e}"
    print("CER score with tf:", cer_scores_previous)

    end_time = time.time()
    print(f"\nEvaluation took {(end_time - start_time) / 60:.2f} minutes to execute.")

    file_content = [
        f"corpus_bleu_score = {corpus_bleu_scores}",
        f"sacre_blue_score = {sacre_blue}",
        f"rouge_scores = {rouge_scores}",
        f"wer_scores = {wer_scores}",
        f"cer_scores = {cer_scores}",
        "\n--- Teacher Forcing Scores ---",
        f"corpus_bleu_score_with_tf = {corpus_bleu_scores_previous}",
        f"sacre_blue_score_with_tf = {sacre_blue_previous}",
        f"rouge_scores_with_tf = {rouge_scores_previous}",
        f"wer_scores_with_tf = {wer_scores_previous}",
        f"cer_scores_with_tf = {cer_scores_previous}",
    ]

    with open(score_results, "w") as file_results:  # Overwrite instead of append
        for line in file_content:
            file_results.write(str(line) + "\n")


if __name__ == "__main__":
    batch_size = 1
    args = get_config("eval_decoding")
    test_input = args["test_input"]
    train_input = args["train_input"]
    print(f"Train input type was: {train_input}")
    print(f"Test input type is: {test_input}")

    if not args["config_path"] or not os.path.exists(args["config_path"]):
        raise FileNotFoundError(
            f"Training config file not found: {args['config_path']}"
        )
    with open(args["config_path"]) as f:
        training_config = json.load(f)

    subject_choice = training_config["subjects"]
    eeg_type_choice = training_config["eeg_type"]
    bands_choice = training_config["eeg_bands"]
    task_name = training_config["task_name"]
    model_name = training_config["model_name"]

    print(f"[INFO] Evaluating model: {model_name} from task: {task_name}")
    print(
        f"[INFO] Subjects: {subject_choice}, EEG Type: {eeg_type_choice}, Bands: {bands_choice}"
    )

    dataset_setting = "unique_sent"

    results_dir = "./results"
    score_dir = "./score_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(args["checkpoint_path"]))[0]
    eval_suffix = f"eval_train-{train_input}_test-{test_input}"

    output_all_results_path = os.path.join(
        results_dir, f"{base_filename}_{eval_suffix}_results.txt"
    )
    score_results_path = os.path.join(
        score_dir, f"{base_filename}_{eval_suffix}_scores.txt"
    )

    print(f"[INFO] Saving all results to: {output_all_results_path}")
    print(f"[INFO] Saving score summary to: {score_results_path}")

    seed_val = 20
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if torch.cuda.is_available():
        dev = args["cuda"]
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f"[INFO] Using device {dev}")

    whole_dataset_dicts = []
    if "task1" in task_name:
        dataset_path_task1 = "/DATA/deep_learning/Aman/EEG-To-Text/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle"
        if os.path.exists(dataset_path_task1):
            with open(dataset_path_task1, "rb") as handle:
                whole_dataset_dicts.append(pickle.load(handle))
        else:
            print(f"Warning: Eval Task 1 dataset not found at {dataset_path_task1}")
    if "task2" in task_name:
        dataset_path_task2 = "/DATA/deep_learning/Aman/EEG-To-Text/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle"
        if os.path.exists(dataset_path_task2):
            with open(dataset_path_task2, "rb") as handle:
                whole_dataset_dicts.append(pickle.load(handle))
        else:
            print(f"Warning: Eval Task 2 dataset not found at {dataset_path_task2}")
    if "task3" in task_name:
        dataset_path_task3 = "/DATA/deep_learning/Aman/EEG-To-Text/dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle"
        if os.path.exists(dataset_path_task3):
            with open(dataset_path_task3, "rb") as handle:
                whole_dataset_dicts.append(pickle.load(handle))
        else:
            print(f"Warning: Eval Task 3 dataset not found at {dataset_path_task3}")
    if "taskNRv2" in task_name:
        dataset_path_taskNRv2 = "/DATA/deep_learning/Aman/EEG-To-Text/dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle"
        if os.path.exists(dataset_path_taskNRv2):
            with open(dataset_path_taskNRv2, "rb") as handle:
                whole_dataset_dicts.append(pickle.load(handle))
        else:
            print(
                f"Warning: Eval Task NRv2 dataset not found at {dataset_path_taskNRv2}"
            )

    if not whole_dataset_dicts:
        raise FileNotFoundError(
            "No dataset pickle files were found or loaded for evaluation."
        )

    if model_name in ["BrainTranslator", "BrainTranslatorNaive"]:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    elif model_name == "PegasusTranslator":
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    elif model_name == "T5Translator":
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
    else:
        raise ValueError(f"Unsupported model_name for tokenizer: {model_name}")

    test_set = ZuCo_dataset(
        whole_dataset_dicts,
        "test",
        tokenizer,
        subject=subject_choice,
        eeg_type=eeg_type_choice,
        bands=bands_choice,
        setting=dataset_setting,
        test_input=test_input,
    )

    print("[INFO] Test set size: ", len(test_set))
    if len(test_set) == 0:
        raise ValueError("Test set is empty. Check dataset construction and filtering.")

    test_dataloader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    dataloaders = {"test": test_dataloader}

    checkpoint_path = args["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    if model_name == "BrainTranslator":
        pretrained = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        model = BrainTranslator(
            pretrained,
            in_feature=105 * len(bands_choice),
            decoder_embedding_size=1024,
            additional_encoder_nhead=8,
            additional_encoder_dim_feedforward=2048,
        )
    elif model_name == "BrainTranslatorNaive":
        pretrained = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        model = BrainTranslatorNaive(
            pretrained,
            in_feature=105 * len(bands_choice),
            decoder_embedding_size=1024,
            additional_encoder_nhead=8,
            additional_encoder_dim_feedforward=2048,
        )
    elif model_name == "PegasusTranslator":
        pretrained = PegasusForConditionalGeneration.from_pretrained(
            "google/pegasus-xsum"
        )
        model = BrainTranslator(
            pretrained,
            in_feature=105 * len(bands_choice),
            decoder_embedding_size=1024,
            additional_encoder_nhead=8,
            additional_encoder_dim_feedforward=2048,
        )
    elif model_name == "T5Translator":
        pretrained = T5ForConditionalGeneration.from_pretrained("t5-large")
        model = T5Translator(
            pretrained,
            in_feature=105 * len(bands_choice),
            decoder_embedding_size=1024,
            additional_encoder_nhead=8,
            additional_encoder_dim_feedforward=2048,
        )
    else:
        raise ValueError(f"Evaluation script does not support model_name: {model_name}")

    state_dict = torch.load(checkpoint_path, map_location=device)

    # Check if the state_dict keys start with 'module.'
    is_data_parallel = all(k.startswith("module.") for k in state_dict.keys())

    if is_data_parallel:
        print("[INFO] Loading weights from DataParallel checkpoint.")
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        print("[INFO] Loading weights from standard checkpoint.")
        model.load_state_dict(state_dict)

    model.to(device)

    # If evaluation should run on multiple GPUs (less common for eval)
    # cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    # if torch.cuda.is_available() and cuda_visible_devices and len(cuda_visible_devices.split(',')) > 1:
    #      eval_device_ids = [int(x) for x in cuda_visible_devices.split(',')]
    #      print(f"[INFO] Using DataParallel for evaluation on devices: {eval_device_ids}")
    #      model = nn.DataParallel(model, device_ids=eval_device_ids)
    # else:
    #      print(f"[INFO] Using single device for evaluation: {device}")

    # Wrap model in DataParallel AFTER loading state_dict if evaluating on multiple GPUs
    # This example assumes single GPU evaluation as set by --cuda argument for simplicity
    # If you used DataParallel during training, model.module.generate needs to be called
    if isinstance(model, nn.DataParallel):
        print("[INFO] Model is wrapped in DataParallel.")
    else:
        print("[INFO] Model is not wrapped in DataParallel.")
        # Need to adjust the generate call in eval_model if the loaded model wasn't DP
        # The current eval_model assumes model *was* DP trained and uses model.module
        # Let's modify eval_model to handle both cases - check if it has 'module' attribute

    criterion = nn.CrossEntropyLoss()

    eval_model(
        dataloaders,
        device,
        tokenizer,
        criterion,
        model,
        output_all_results_path=output_all_results_path,
        score_results=score_results_path,
    )

    print("--- Evaluation Finished ---")
