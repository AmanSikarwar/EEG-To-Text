import copy
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    PegasusConfig,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

from config import get_config
from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive, T5Translator


def train_model(
    dataloaders,
    device,
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
    checkpoint_path_best="./checkpoints/decoding/best/temp_decoding.pt",
    checkpoint_path_last="./checkpoints/decoding/last/temp_decoding.pt",
    dataset_sizes=None,
):
    since = time.time()
    if dataset_sizes is None:
        raise ValueError("dataset_sizes dictionary must be provided")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    underlying_model = model.module if isinstance(model, nn.DataParallel) else model

    tokenizer = None
    if hasattr(underlying_model, "tokenizer"):
        tokenizer = underlying_model.tokenizer
        print("[INFO] Found tokenizer directly on the model instance.")
    elif hasattr(underlying_model, "pretrained") and hasattr(
        underlying_model.pretrained, "tokenizer"
    ):
        tokenizer = underlying_model.pretrained.tokenizer
        print("[INFO] Found tokenizer on the 'pretrained' attribute.")
    elif (
        hasattr(underlying_model, "pretrained")
        and hasattr(underlying_model.pretrained, "config")
        and hasattr(underlying_model.pretrained.config, "_name_or_path")
    ):
        model_config_path = underlying_model.pretrained.config._name_or_path
        print(f"[INFO] Attempting to load tokenizer from config: {model_config_path}")
        try:
            if "t5" in model_config_path.lower():
                tokenizer = T5Tokenizer.from_pretrained(model_config_path)
            elif "pegasus" in model_config_path.lower():
                tokenizer = PegasusTokenizer.from_pretrained(model_config_path)
            elif "bart" in model_config_path.lower():
                tokenizer = BartTokenizer.from_pretrained(model_config_path)
            else:
                print(
                    f"Warning: Unknown model type '{model_config_path}' in config. Attempting BART tokenizer."
                )
                tokenizer = BartTokenizer.from_pretrained(model_config_path)
        except Exception as e:
            print(
                f"Could not load tokenizer from pretrained config '{model_config_path}': {e}"
            )
            tokenizer = None
    else:
        print(
            "[INFO] Could not find tokenizer attribute directly or via pretrained layer/config."
        )

    if tokenizer is None:
        print(
            "ERROR: Could not determine the tokenizer for the model. Please check model implementation or provide explicitly."
        )
        print("Warning: Defaulting to BART tokenizer.")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    print(f"[INFO] Using tokenizer: {type(tokenizer).__name__}")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "dev"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for (
                input_embeddings,
                seq_len,
                input_masks,
                input_mask_invert,
                target_ids,
                target_mask,
                sentiment_labels,
            ) in tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch}"):
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)

                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    seq2seqLMoutput = model(
                        input_embeddings_batch,
                        input_masks_batch,
                        input_mask_invert_batch,
                        target_ids_batch,
                    )
                    if isinstance(seq2seqLMoutput.loss, torch.Tensor):
                        loss = seq2seqLMoutput.loss.mean()
                    else:
                        loss = seq2seqLMoutput.loss

                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()

                running_loss += loss.item() * input_embeddings_batch.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f}")

            if phase == "dev" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                )
                torch.save(best_model_wts, checkpoint_path_best)
                print(f"update best on dev checkpoint: {checkpoint_path_best}")

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val loss: {best_loss:4f}")
    last_model_wts = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(last_model_wts, checkpoint_path_last)
    print(f"update last checkpoint: {checkpoint_path_last}")

    model.load_state_dict(torch.load(checkpoint_path_best, map_location=device))
    return model


def show_require_grad_layers(model):
    print("\n require_grad layers:")
    model_to_inspect = model.module if isinstance(model, nn.DataParallel) else model
    for name, param in model_to_inspect.named_parameters():
        if param.requires_grad:
            print(f"  {name}")


if __name__ == "__main__":
    args = get_config("train_decoding")

    dataset_setting = "unique_sent"
    num_epochs_step1 = args["num_epoch_step1"]
    num_epochs_step2 = args["num_epoch_step2"]
    step1_lr = args["learning_rate_step1"]
    step2_lr = args["learning_rate_step2"]
    batch_size = args["batch_size"]
    model_name = args["model_name"]
    task_name = args["task_name"]
    train_input = args["train_input"]
    save_path = args["save_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    skip_step_one = args["skip_step_one"]
    load_step1_checkpoint = args["load_step1_checkpoint"]
    use_random_init = args["use_random_init"]
    default_device_ids = [0, 1]

    print(f"[INFO] using model: {model_name}")

    if skip_step_one:
        save_name_prefix = f"{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}"
    else:
        save_name_prefix = f"{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}"

    if use_random_init:
        save_name = "randinit_" + save_name_prefix
    else:
        save_name = save_name_prefix

    save_path_best = os.path.join(save_path, "best")
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)
    output_checkpoint_name_best = os.path.join(save_path_best, f"{save_name}.pt")

    save_path_last = os.path.join(save_path, "last")
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)
    output_checkpoint_name_last = os.path.join(save_path_last, f"{save_name}.pt")

    subject_choice = args["subjects"]
    eeg_type_choice = args["eeg_type"]
    bands_choice = args["eeg_bands"]

    print(
        f"[INFO] Subject: {subject_choice}, EEG Type: {eeg_type_choice}, Bands: {bands_choice}"
    )

    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device_ids = []
    if torch.cuda.is_available():
        specified_cuda_arg = args["cuda"]
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")

        if cuda_visible_devices:
            all_gpu_indices = [int(x) for x in cuda_visible_devices.split(",")]
            try:
                primary_device_index_rel = int(specified_cuda_arg.split(":")[-1])
                if primary_device_index_rel < len(all_gpu_indices):
                    primary_device_abs = all_gpu_indices[primary_device_index_rel]
                    dev = f"cuda:{primary_device_abs}"
                    device_ids = list(range(len(all_gpu_indices)))
                    print(
                        f"[INFO] Using CUDA. Primary device (abs): {dev}. Visible devices (abs): {all_gpu_indices}. DataParallel on relative devices: {device_ids}"
                    )
                else:
                    print(
                        f"Warning: Specified device {specified_cuda_arg} index out of range for visible devices {all_gpu_indices}. Using first visible device."
                    )
                    dev = f"cuda:{all_gpu_indices[0]}"
                    device_ids = list(range(len(all_gpu_indices)))
            except (ValueError, IndexError):
                print(
                    f"Warning: Could not parse {specified_cuda_arg} or map to visible devices. Using first visible device {all_gpu_indices[0]}."
                )
                dev = f"cuda:{all_gpu_indices[0]}"
                device_ids = list(range(len(all_gpu_indices)))

        else:
            dev = specified_cuda_arg
            device_ids = [int(dev.split(":")[-1])]
            print(
                f"[INFO] Using CUDA device specified by --cuda: {dev}. CUDA_VISIBLE_DEVICES not set."
            )

    else:
        dev = "cpu"
        device_ids = []
        print("[INFO] Using CPU")

    device = torch.device(dev)

    whole_dataset_dicts = []
    base_data_path = "/DATA/deep_learning/Aman/EEG-To-Text/dataset/ZuCo/"
    dataset_paths = {
        "task1": os.path.join(
            base_data_path, "task1-SR/pickle/task1-SR-dataset.pickle"
        ),
        "task2": os.path.join(
            base_data_path, "task2-NR/pickle/task2-NR-dataset.pickle"
        ),
        "task3": os.path.join(
            base_data_path, "task3-TSR/pickle/task3-TSR-dataset.pickle"
        ),
        "taskNRv2": os.path.join(
            base_data_path, "task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle"
        ),
    }

    tasks_to_load = []
    if "task1" in task_name:
        tasks_to_load.append("task1")
    if "task2" in task_name:
        tasks_to_load.append("task2")
    if "task3" in task_name:
        tasks_to_load.append("task3")
    if "taskNRv2" in task_name:
        tasks_to_load.append("taskNRv2")

    for task_key in tasks_to_load:
        d_path = dataset_paths.get(task_key)
        if d_path and os.path.exists(d_path):
            print(f"Loading dataset: {d_path}")
            with open(d_path, "rb") as handle:
                whole_dataset_dicts.append(pickle.load(handle))
        else:
            print(f"Warning: Dataset for {task_key} not found at {d_path}")

    if not whole_dataset_dicts:
        raise FileNotFoundError(
            "No dataset pickle files were found or loaded. Please check paths."
        )

    cfg_dir = "./config/decoding/"
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
    with open(os.path.join(cfg_dir, f"{save_name}.json"), "w") as out_config:
        json.dump(args, out_config, indent=4)

    if model_name in ["BrainTranslator", "BrainTranslatorNaive"]:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    elif model_name == "PegasusTranslator":
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    elif model_name == "T5Translator":
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
    else:
        raise ValueError(f"Unsupported model_name for tokenizer: {model_name}")

    train_set = ZuCo_dataset(
        whole_dataset_dicts,
        "train",
        tokenizer,
        subject=subject_choice,
        eeg_type=eeg_type_choice,
        bands=bands_choice,
        setting=dataset_setting,
        test_input=train_input,
    )
    dev_set = ZuCo_dataset(
        whole_dataset_dicts,
        "dev",
        tokenizer,
        subject=subject_choice,
        eeg_type=eeg_type_choice,
        bands=bands_choice,
        setting=dataset_setting,
        test_input=train_input,
    )

    dataset_sizes = {"train": len(train_set), "dev": len(dev_set)}
    print("[INFO]train_set size: ", len(train_set))
    print("[INFO]dev_set size: ", len(dev_set))

    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        dev_set,
        batch_size=batch_size // 2 if batch_size > 1 else 1,
        shuffle=False,
        num_workers=4,
    )
    dataloaders = {"train": train_dataloader, "dev": val_dataloader}

    if model_name == "BrainTranslator":
        if use_random_init:
            config = BartConfig.from_pretrained("facebook/bart-large")
            pretrained = BartForConditionalGeneration(config)
        else:
            pretrained = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-large"
            )
        model = BrainTranslator(
            pretrained,
            in_feature=105 * len(bands_choice),
            decoder_embedding_size=1024,
            additional_encoder_nhead=8,
            additional_encoder_dim_feedforward=2048,
        )

    elif model_name == "BrainTranslatorNaive":
        if use_random_init:
            config = BartConfig.from_pretrained("facebook/bart-large")
            pretrained = BartForConditionalGeneration(config)
        else:
            pretrained = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-large"
            )
        model = BrainTranslatorNaive(
            pretrained,
            in_feature=105 * len(bands_choice),
            decoder_embedding_size=1024,
            additional_encoder_nhead=8,
            additional_encoder_dim_feedforward=2048,
        )

    elif model_name == "PegasusTranslator":
        if use_random_init:
            config = PegasusConfig.from_pretrained("google/pegasus-xsum")
            pretrained = PegasusForConditionalGeneration(config)
        else:
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
        if use_random_init:
            config = T5Config.from_pretrained("t5-large")
            pretrained = T5ForConditionalGeneration(config)
        else:
            pretrained = T5ForConditionalGeneration.from_pretrained("t5-large")
        model = T5Translator(
            pretrained,
            in_feature=105 * len(bands_choice),
            decoder_embedding_size=1024,
            additional_encoder_nhead=8,
            additional_encoder_dim_feedforward=2048,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.to(device)

    use_data_parallel = torch.cuda.is_available() and len(device_ids) > 1
    if use_data_parallel:
        print(f"[INFO] Using DataParallel across devices: {device_ids}")
        relative_device_ids = list(range(len(device_ids)))
        model = torch.nn.DataParallel(model, device_ids=relative_device_ids)
        print(
            f"[INFO] DataParallel active on relative device IDs: {relative_device_ids}"
        )
    elif torch.cuda.is_available():
        print(f"[INFO] Using single GPU: {device}")
    else:
        print("[INFO] Using CPU, DataParallel not applicable.")

    criterion = nn.CrossEntropyLoss()

    if not skip_step_one:
        print("\n=== Starting Step 1 Training ===")
        model_to_freeze = model.module if use_data_parallel else model
        if model_name in [
            "BrainTranslator",
            "BrainTranslatorNaive",
            "PegasusTranslator",
            "T5Translator",
        ]:
            print("Freezing layers for Step 1...")
            for name, param in model_to_freeze.named_parameters():
                freeze = True
                if "pretrained" in name:
                    if any(
                        sub in name
                        for sub in [
                            "shared",
                            "embed_tokens",
                            "embed_positions",
                            "encoder.layers.0",
                            "encoder.block.0",
                        ]
                    ):
                        freeze = False
                elif "fc1" in name or "additional_encoder" in name:
                    freeze = False

                if freeze:
                    param.requires_grad = False

        show_require_grad_layers(model)
        optimizer_step1 = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=step1_lr,
            weight_decay=0.01,
        )
        total_steps_s1 = len(dataloaders["train"]) * num_epochs_step1
        num_warmup_steps_s1 = int(0.1 * total_steps_s1)
        scheduler_step1 = get_linear_schedule_with_warmup(
            optimizer_step1,
            num_warmup_steps=num_warmup_steps_s1,
            num_training_steps=total_steps_s1,
        )

        model = train_model(
            dataloaders,
            device,
            model,
            criterion,
            optimizer_step1,
            scheduler_step1,
            num_epochs=num_epochs_step1,
            checkpoint_path_best=output_checkpoint_name_best,
            checkpoint_path_last=output_checkpoint_name_last,
            dataset_sizes=dataset_sizes,
        )

    elif load_step1_checkpoint:
        stepone_checkpoint = output_checkpoint_name_best
        if os.path.exists(stepone_checkpoint):
            print(f"Skipping step one, loading checkpoint: {stepone_checkpoint}")
            model_to_load = model.module if use_data_parallel else model
            model_to_load.load_state_dict(
                torch.load(stepone_checkpoint, map_location=device)
            )
        else:
            print(
                f"Warning: Load step1 checkpoint requested, but file not found: {stepone_checkpoint}. Starting step 2 from scratch."
            )
    else:
        print("Skipping step one, starting step 2 from scratch.")

    print("\n=== Starting Step 2 Training ===")
    print("Unfreezing all layers for Step 2...")
    for param in model.parameters():
        param.requires_grad = True

    show_require_grad_layers(model)
    optimizer_step2 = AdamW(model.parameters(), lr=step2_lr, weight_decay=0.01)
    total_steps_s2 = len(dataloaders["train"]) * num_epochs_step2
    num_warmup_steps_s2 = int(0.1 * total_steps_s2)
    scheduler_step2 = get_linear_schedule_with_warmup(
        optimizer_step2,
        num_warmup_steps=num_warmup_steps_s2,
        num_training_steps=total_steps_s2,
    )

    trained_model = train_model(
        dataloaders,
        device,
        model,
        criterion,
        optimizer_step2,
        scheduler_step2,
        num_epochs=num_epochs_step2,
        checkpoint_path_best=output_checkpoint_name_best,
        checkpoint_path_last=output_checkpoint_name_last,
        dataset_sizes=dataset_sizes,
    )

    print("--- Training Finished ---")
