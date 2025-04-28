#!/bin/bash

SENDER_EMAIL="pranavpashirbhate16@gmail.com"
RECIPIENT_EMAIL="amansik.1910@gmail.com"
LOG_DIR="./run_logs"
TRAIN_LOG="${LOG_DIR}/train_decoding_$(date +%Y%m%d_%H%M%S).log"
EVAL_LOG="${LOG_DIR}/eval_decoding_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="${LOG_DIR}/status.txt"


TRAIN_SCRIPT_ID="Train T5 EEG-to-Text (2-Step)"
EVAL_SCRIPT_ID="Eval T5 EEG-to-Text"

mkdir -p "$LOG_DIR"
rm -f "$STATUS_FILE"

OVERALL_STATUS="SUCCESS"
START_TIME=$(date)

TASK_NAME="task1_task2_taskNRv2"
MODEL_NAME="T5Translator"
BATCH_SIZE=16
NUM_EPOCH_S1=20
NUM_EPOCH_S2=30
LR1="0.0001"
LR2="0.000005"
DATASET_SETTING="unique_sent"
TRAIN_INPUT="EEG"

CHECKPOINT_BASE_NAME="${TASK_NAME}_finetune_${MODEL_NAME}_2steptraining_b${BATCH_SIZE}_${NUM_EPOCH_S1}_${NUM_EPOCH_S2}_${LR1}_${LR2}_${DATASET_SETTING}_${TRAIN_INPUT}"

BEST_CHECKPOINT_PATH="./checkpoints/decoding/best/${CHECKPOINT_BASE_NAME}.pt"
CONFIG_PATH="./config/decoding/${CHECKPOINT_BASE_NAME}.json"


cleanup_and_notify() {
    EXIT_CODE=$?
    END_TIME=$(date)
    FINAL_MESSAGE_BODY="Experiment Run Summary:\n"
    FINAL_MESSAGE_BODY+="--------------------------\n"
    FINAL_MESSAGE_BODY+="Start Time: ${START_TIME}\n"
    FINAL_MESSAGE_BODY+="End Time:   ${END_TIME}\n"
    FINAL_MESSAGE_BODY+="Checkpoint Base Name: ${CHECKPOINT_BASE_NAME}\n"
    FINAL_MESSAGE_BODY+="--------------------------\n\n"

    if [[ "$EXIT_CODE" -ne 0 && "$OVERALL_STATUS" == "SUCCESS" ]]; then
        OVERALL_STATUS="INTERRUPTED"
        echo "Script interrupted (Exit Code: ${EXIT_CODE})." >> "$STATUS_FILE"
    elif [[ "$EXIT_CODE" -ne 0 && "$OVERALL_STATUS" == "FAILURE" ]]; then
        echo "Script failed (Exit Code: ${EXIT_CODE})." >> "$STATUS_FILE"
    elif [[ "$EXIT_CODE" -eq 0 && "$OVERALL_STATUS" != "SUCCESS" ]]; then
        echo "Script exited cleanly but recorded internal failure." >> "$STATUS_FILE"
        OVERALL_STATUS="FAILURE"
    fi

    SUBJECT="[EEG-To-Text Monitor] Run ${OVERALL_STATUS}: ${CHECKPOINT_BASE_NAME}"

    if [[ "$OVERALL_STATUS" == "FAILURE" || "$OVERALL_STATUS" == "INTERRUPTED" ]]; then
        FINAL_MESSAGE_BODY+="Status: ${OVERALL_STATUS}\n\n"
        if [[ -f "$STATUS_FILE" ]]; then
            STATUS_CONTENT=$(cat "$STATUS_FILE")
            FINAL_MESSAGE_BODY+="Reason/Last Stage:\n${STATUS_CONTENT}\n\n"
        fi
            FINAL_MESSAGE_BODY+="Please check the log files for details:\n"
            FINAL_MESSAGE_BODY+="- Training Log: ${TRAIN_LOG}\n"
            FINAL_MESSAGE_BODY+="- Evaluation Log: ${EVAL_LOG}\n"
    else
        FINAL_MESSAGE_BODY+="Status: ${OVERALL_STATUS}\n\n"
        FINAL_MESSAGE_BODY+="Training and evaluation completed successfully.\n"
        FINAL_MESSAGE_BODY+="Best Checkpoint: ${BEST_CHECKPOINT_PATH}\n"
        FINAL_MESSAGE_BODY+="Training Log: ${TRAIN_LOG}\n"
        FINAL_MESSAGE_BODY+="Evaluation Log: ${EVAL_LOG}\n"
    fi

    TEMP_BODY_FILE=$(mktemp)
    if [[ -z "$TEMP_BODY_FILE" ]]; then
        echo "ERROR: Could not create temporary file for email body." >&2
        echo "Run ${OVERALL_STATUS}. Failed to create temp file for body." | yagmail -u "$SENDER_EMAIL" -to "$RECIPIENT_EMAIL" -s "$SUBJECT" -c -
        return 1
    fi

    printf "%b" "$FINAL_MESSAGE_BODY" > "$TEMP_BODY_FILE"

    echo "Sending notification email to ${RECIPIENT_EMAIL}..."
    yagmail -u "$SENDER_EMAIL" -to "$RECIPIENT_EMAIL" -s "$SUBJECT" -c "$(< "$TEMP_BODY_FILE")"

    if [[ $? -ne 0 ]]; then
        echo "WARNING: yagmail command failed to send notification." >&2
    else
        echo "Notification sent."
    fi

    rm -f "$TEMP_BODY_FILE"
}

trap cleanup_and_notify EXIT INT TERM

# echo "--- Starting Training Phase: ${TRAIN_SCRIPT_ID} ---"
# echo "Logging to ${TRAIN_LOG}"
# echo "Checkpoint Base Name: ${CHECKPOINT_BASE_NAME}"

# echo "Running Training Command (2-Step)..."
# # Ensure CUDA_VISIBLE_DEVICES matches the number of devices expected by DataParallel (e.g., 2 if device_ids=[0,1])
# CUDA_VISIBLE_DEVICES=0 python train_decoding2.py \
#     --model_name ${MODEL_NAME} \
#     --task_name ${TASK_NAME} \
#     --two_step \
#     --pretrained \
#     --not_load_step1_checkpoint \
#     --num_epoch_step1 ${NUM_EPOCH_S1} \
#     --num_epoch_step2 ${NUM_EPOCH_S2} \
#     --train_input ${TRAIN_INPUT} \
#     -lr1 ${LR1} \
#     -lr2 ${LR2} \
#     -b ${BATCH_SIZE} \
#     -s ./checkpoints/decoding \
#     --cuda cuda:0 \
#     > >(tee -a "${TRAIN_LOG}") 2>&1

# if [[ $? -ne 0 ]]; then
#     echo "ERROR: ${TRAIN_SCRIPT_ID} - Training failed." | tee -a "$STATUS_FILE"
#     OVERALL_STATUS="FAILURE"
#     exit 1
# fi
# echo "Training finished."

# echo "--- Training Phase Successfully Completed ---"


echo "--- Starting Evaluation Phase: ${EVAL_SCRIPT_ID} ---"
echo "Logging to ${EVAL_LOG}"
echo "Evaluating Checkpoint: ${BEST_CHECKPOINT_PATH}"
echo "Using Config: ${CONFIG_PATH}"

if [[ ! -f "$BEST_CHECKPOINT_PATH" ]]; then
    echo "ERROR: Evaluation failed. Checkpoint file not found: $BEST_CHECKPOINT_PATH" | tee -a "$STATUS_FILE"
    OVERALL_STATUS="FAILURE"
    exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Evaluation failed. Config file not found: $CONFIG_PATH" | tee -a "$STATUS_FILE"
    OVERALL_STATUS="FAILURE"
    exit 1
fi


echo "Running Evaluation Command 1 (Test EEG)..."
CUDA_VISIBLE_DEVICES=0 python eval_decoding2.py \
    --checkpoint_path "${BEST_CHECKPOINT_PATH}" \
    --config_path "${CONFIG_PATH}" \
    --test_input EEG \
    --train_input ${TRAIN_INPUT} \
    -cuda cuda:0 > >(tee -a "${EVAL_LOG}") 2>&1

if [[ $? -ne 0 ]]; then
    echo "ERROR: ${EVAL_SCRIPT_ID} - Command 1 (Test EEG) failed." | tee -a "$STATUS_FILE"
    OVERALL_STATUS="FAILURE"
    exit 1
fi
echo "Evaluation Command 1 finished."


echo "Running Evaluation Command 2 (Test Noise)..."
CUDA_VISIBLE_DEVICES=0 python eval_decoding2.py \
    --checkpoint_path "${BEST_CHECKPOINT_PATH}" \
    --config_path "${CONFIG_PATH}" \
    --test_input noise \
    --train_input ${TRAIN_INPUT} \
    -cuda cuda:0 >> >(tee -a "${EVAL_LOG}") 2>&1

if [[ $? -ne 0 ]]; then
    echo "ERROR: ${EVAL_SCRIPT_ID} - Command 2 (Test Noise) failed." | tee -a "$STATUS_FILE"
    OVERALL_STATUS="FAILURE"
    exit 1
fi
echo "Evaluation Command 2 finished."

echo "--- Evaluation Phase Successfully Completed ---"

exit 0