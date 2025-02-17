GPU_NUMBER=0
MODEL_NAME='xlm-roberta-base'
TASK='MultiEURLEX'  #https://huggingface.co/datasets/coastalcph/multi_eurlex
BATCH_SIZE=16
ACCUMULATION_STEPS=2
SPLITSEED=1
MODELSEED=42
LANG='da' #da  de fr  en ; it es pl ro, nl el hu pt, cs sv bg fi, sk lt hr sl, et lv mt
RUN_FILE='/TemporalLearning-MoTE/mote.py'
OUTPUT_DIR=/HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/${LANG}/MoE+router+concatenate_diff
MAX_SEQ_LENGTH=512
LEARNING_RATE=3e-5
NUM_EPOCHS=10
TRAIN_SPLIT_RATIO=0.3
TRAIN_BATCH_SIZE=${BATCH_SIZE}
EVAL_BATCH_SIZE=${BATCH_SIZE}

# Running the classification script
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} \
    --seed ${MODELSEED} \
    --model_name_or_path /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/${LANG}/eval_test_two_time/checkpoint-820 \
    --eval_split_ratio ${TRAIN_SPLIT_RATIO} \
    --train_split_ratio ${TRAIN_SPLIT_RATIO} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --dataset_name 'multi_eurlex' \
    --dataset_config_name ${LANG} \
    --label_column_name 'labels' \
    --text_column_names 'text' \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --do_train --do_predict \

