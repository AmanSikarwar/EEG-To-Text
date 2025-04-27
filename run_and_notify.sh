#!/bin/bash

# --- Configuration ---
SENDER_EMAIL="pranavpashirbhate16@gmail.com" # The email registered with yagmail
RECIPIENT_EMAIL="amansik.1910@gmail.com"
LOG_DIR="./run_logs"
TRAIN_LOG="${LOG_DIR}/train_decoding.log"
EVAL_LOG="${LOG_DIR}/eval_decoding.log"
STATUS_FILE="${LOG_DIR}/status.txt" # To store failure reason

# --- Script Names (for logging/messages) ---
TRAIN_SCRIPT_ID="Train T5 EEG-to-Text"
EVAL_SCRIPT_ID="Eval T5 EEG-to-Text"

# --- Ensure log directory exists ---
mkdir -p "$LOG_DIR"
# --- Clean up previous status ---
rm -f "$STATUS_FILE"

# --- Variables ---
OVERALL_STATUS="SUCCESS"
START_TIME=$(date)

# --- Cleanup and Notification Function ---
# This function is called when the script exits (normally, via error, or interruption)
cleanup_and_notify() {
    EXIT_CODE=$? # Capture the script's final exit code
    END_TIME=$(date)
    FINAL_MESSAGE_BODY="Experiment Run Summary:\n"
    FINAL_MESSAGE_BODY+="--------------------------\n"
    FINAL_MESSAGE_BODY+="Start Time: ${START_TIME}\n"
    FINAL_MESSAGE_BODY+="End Time:   ${END_TIME}\n"
    FINAL_MESSAGE_BODY+="--------------------------\n\n"

    # Determine final status based on exit code and recorded status
    if [[ "$EXIT_CODE" -ne 0 && "$OVERALL_STATUS" == "SUCCESS" ]]; then
        # Script exited non-zero but status wasn't set to FAILURE -> Interruption
        OVERALL_STATUS="INTERRUPTED"
        echo "Script interrupted (Exit Code: ${EXIT_CODE})." >> "$STATUS_FILE"
    elif [[ "$EXIT_CODE" -ne 0 && "$OVERALL_STATUS" == "FAILURE" ]]; then
        # Script exited non-zero and status was already FAILURE -> Failure occurred
        echo "Script failed (Exit Code: ${EXIT_CODE})." >> "$STATUS_FILE" # Append exit code info
    elif [[ "$EXIT_CODE" -eq 0 && "$OVERALL_STATUS" != "SUCCESS" ]]; then
         # Script exited cleanly (0), but status was set to FAILURE -> Failure handled internally
         echo "Script exited cleanly but recorded internal failure." >> "$STATUS_FILE" # Should ideally not happen with exit 1 below
         OVERALL_STATUS="FAILURE" # Ensure status is correct
    fi

    SUBJECT="[EEG-To-Text Monitor] Run ${OVERALL_STATUS}"

    if [[ "$OVERALL_STATUS" == "FAILURE" || "$OVERALL_STATUS" == "INTERRUPTED" ]]; then
        FINAL_MESSAGE_BODY+="Status: ${OVERALL_STATUS}\n\n"
        if [[ -f "$STATUS_FILE" ]]; then
             FINAL_MESSAGE_BODY+="Reason/Last Stage:\n$(cat "$STATUS_FILE")\n\n"
        fi
         FINAL_MESSAGE_BODY+="Please check the log files for details:\n"
         FINAL_MESSAGE_BODY+="- Training Log: ${TRAIN_LOG}\n"
         FINAL_MESSAGE_BODY+="- Evaluation Log: ${EVAL_LOG}\n"
    else
        FINAL_MESSAGE_BODY+="Status: ${OVERALL_STATUS}\n\n"
        FINAL_MESSAGE_BODY+="Training and evaluation completed successfully.\n"
        FINAL_MESSAGE_BODY+="Training Log: ${TRAIN_LOG}\n"
        FINAL_MESSAGE_BODY+="Evaluation Log: ${EVAL_LOG}\n"
    fi

    echo "Sending notification email to ${RECIPIENT_EMAIL}..."
    # Use yagmail CLI - explicitly provide sender using -u
    yagmail -u "$SENDER_EMAIL" -to "$RECIPIENT_EMAIL" -s "$SUBJECT" -c "$FINAL_MESSAGE_BODY"

    # Optional: Attach logs if needed (can make emails large) - also add -u here if using
    # yagmail -u "$SENDER_EMAIL" -to "$RECIPIENT_EMAIL" -s "$SUBJECT" -c "$FINAL_MESSAGE_BODY" -a "$TRAIN_LOG" "$EVAL_LOG"

    # Add a basic check for the yagmail command itself (optional but good practice)
    if [[ $? -ne 0 ]]; then
        echo "WARNING: yagmail command failed to send notification." >&2 # Print to stderr
    else
        echo "Notification sent."
    fi
}

# --- Trap Exit, Interrupt (Ctrl+C), and Termination Signals ---
# The 'EXIT' trap ensures cleanup_and_notify runs regardless of how the script ends
trap cleanup_and_notify EXIT INT TERM

# --- Training Phase ---
# echo "--- Starting Training Phase: ${TRAIN_SCRIPT_ID} ---"
# echo "Logging to ${TRAIN_LOG}"

# # Command 1
# echo "Running Training Command 1..."
# CUDA_VISIBLE_DEVICES=2,3 python3 train_decoding.py --model_name T5Translator \
#     --task_name task1_task2_taskNRv2 \
#     --one_step \
#     --pretrained \
#     --not_load_step1_checkpoint \
#     --num_epoch_step1 20 \
#     --num_epoch_step2 30 \
#     --train_input EEG \
#     -lr1 0.00002 \
#     -lr2 0.00002 \
#     -b 32 \
#     -s ./checkpoints/decoding > >(tee -a "${TRAIN_LOG}") 2>&1 # Log stdout/stderr

# if [[ $? -ne 0 ]]; then
#     echo "ERROR: ${TRAIN_SCRIPT_ID} - Command 1 failed." | tee -a "$STATUS_FILE"
#     OVERALL_STATUS="FAILURE"
#     exit 1 # Exit immediately upon failure
# fi
# echo "Training Command 1 finished."

# # Command 2
# echo "Running Training Command 2..."
# CUDA_VISIBLE_DEVICES=2,3 python3 train_decoding.py --model_name T5Translator \
#     --task_name task1_task2_task3 \
#     --one_step \
#     --pretrained \
#     --not_load_step1_checkpoint \
#     --num_epoch_step1 20 \
#     --num_epoch_step2 30 \
#     --train_input EEG \
#     -lr1 0.00002 \
#     -lr2 0.00002 \
#     -b 32 \
#     -s ./checkpoints/decoding >> >(tee -a "${TRAIN_LOG}") 2>&1 # Append log stdout/stderr

# if [[ $? -ne 0 ]]; then
#     echo "ERROR: ${TRAIN_SCRIPT_ID} - Command 2 failed." | tee -a "$STATUS_FILE"
#     OVERALL_STATUS="FAILURE"
#     exit 1 # Exit immediately upon failure
# fi
# echo "Training Command 2 finished."

# echo "--- Training Phase Successfully Completed ---"


# --- Evaluation Phase ---
# Only run if training was successful
echo "--- Starting Evaluation Phase: ${EVAL_SCRIPT_ID} ---"
echo "Logging to ${EVAL_LOG}"

# Command 1
echo "Running Evaluation Command 1 (Test EEG)..."
CUDA_VISIBLE_DEVICES=1 python eval_decoding.py \
    --checkpoint_path checkpoints/decoding/best/task1_task2_taskNRv2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.pt \
    --config_path config/decoding/task1_task2_taskNRv2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.json \
    --test_input EEG \
    --train_input EEG \
    -cuda cuda:0 > >(tee -a "${EVAL_LOG}") 2>&1 # Log stdout/stderr

if [[ $? -ne 0 ]]; then
    echo "ERROR: ${EVAL_SCRIPT_ID} - Command 1 (Test EEG) failed." | tee -a "$STATUS_FILE"
    OVERALL_STATUS="FAILURE"
    exit 1 # Exit immediately upon failure
fi
echo "Evaluation Command 1 finished."


# Command 2
echo "Running Evaluation Command 2 (Test Noise)..."
CUDA_VISIBLE_DEVICES=1 python eval_decoding.py \
    --checkpoint_path checkpoints/decoding/best/task1_task2_taskNRv2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.pt \
    --config_path config/decoding/task1_task2_taskNRv2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.json \
    --test_input noise \
    --train_input EEG \
    -cuda cuda:0 >> >(tee -a "${EVAL_LOG}") 2>&1 # Append log stdout/stderr

if [[ $? -ne 0 ]]; then
    echo "ERROR: ${EVAL_SCRIPT_ID} - Command 2 (Test Noise) failed." | tee -a "$STATUS_FILE"
    OVERALL_STATUS="FAILURE"
    exit 1 # Exit immediately upon failure
fi
echo "Evaluation Command 2 finished."

echo "--- Evaluation Phase Successfully Completed ---"

# --- Script finished successfully ---
# The EXIT trap will handle the final success notification
exit 0