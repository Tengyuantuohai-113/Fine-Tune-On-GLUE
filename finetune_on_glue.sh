python -u finetune_on_glue.py \
    --dataset "cola" \
    --model "bert-base-cased" \
    --batch_size 64 \
    --max_length 512 \
    --lr 5e-4 \
    --weight_decay 1e-3 \
    --epochs 10 \
    --output_dir "./finetuned_model" \
    --seed 42 > "fine_tune_bert.log"

