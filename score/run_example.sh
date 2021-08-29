
DATA_DIR="data"


####
# Pre-train SCoRe (CCS + TCS) used for SParC, CoSQL, and SQA tasks
####
LOGDIR="score_logs_checkpoints/augment_all_context_roberta"
rm -r $LOGDIR
mkdir $LOGDIR

CUDA_VISIBLE_DEVICES=0 python -u finetuning_roberta.py --train_corpus "$DATA_DIR/augment_all_context.txt" \
                                                       --eval_corpus "$DATA_DIR/spider_dev_data_v2.txt" \
                                                       --train_eval_corpus "$DATA_DIR/spider_train_data_small_v2.txt" \
                                                       --bert_model roberta-large \
                                                       --output_dir $LOGDIR/ \
                                                       --do_train \
                                                       --do_eval \
                                                       --diff_loss \
                                                       --train_batch_size 12 \
                                                       --max_seq_length 248 \
                                                       --num_train_epochs 15 \
                                                       
# # Use c training if multiple (8) GPUs are available
# LOGDIR="score_logs_checkpoints/augment_all_context_roberta"
# export NGPU=8;
# python -u -m torch.distributed.launch --nproc_per_node=$NGPU finetuning_roberta.py --train_corpus "$DATA_DIR/augment_all_context.txt" \
#                                                        --eval_corpus "$DATA_DIR/spider_dev_data_v2.txt" \
#                                                        --train_eval_corpus "$DATA_DIR/spider_train_data_small_v2.txt" \
#                                                        --bert_model roberta-large \
#                                                        --output_dir $LOGDIR/ \
#                                                        --do_train \
#                                                        --do_eval \
#                                                        --diff_loss \
#                                                        --train_batch_size 12 \
#                                                        --max_seq_length 248 \
#                                                        --num_train_epochs 15 \




# ####
# # Pre-train SCoRe (CCS + TCS + MLM) used for SParC, CoSQL, and SQA tasks
# ####
# LOGDIR="score_logs_checkpoints/augment_all_mlm_context_roberta"
# rm -r $LOGDIR
# mkdir $LOGDIR

# CUDA_VISIBLE_DEVICES=0 python -u finetuning_roberta.py --train_corpus "$DATA_DIR/augment_all_mlm_context.txt" \
#                                                        --eval_corpus "$DATA_DIR/spider_dev_data_v2.txt" \
#                                                        --train_eval_corpus "$DATA_DIR/spider_train_data_small_v2.txt" \
#                                                        --bert_model roberta-large \
#                                                        --output_dir $LOGDIR/ \
#                                                        --do_train \
#                                                        --do_eval \
#                                                        --mlm_loss \
#                                                        --diff_loss \
#                                                        --train_batch_size 12 \
#                                                        --max_seq_length 248 \
#                                                        --num_train_epochs 15 \



                                                       
# ####
# # Pre-train SCoRe (CCS + MLM) used for MWoZ
# ####
# LOGDIR="score_logs_checkpoints/augment_multiwoz_mlm_dst_512_bert_base"
# rm -r $LOGDIR
# mkdir $LOGDIR

# CUDA_VISIBLE_DEVICES=0 python -u finetuning_bert_long.py --train_corpus "$DATA_DIR/augment_multiwoz_mlm_dst_512.txt" \
#                                                        --eval_corpus "$DATA_DIR/spider_dev_data_v2.txt" \
#                                                        --train_eval_corpus "$DATA_DIR/spider_train_data_small_v2.txt" \
#                                                        --bert_model bert-base-uncased \
#                                                        --output_dir $LOGDIR/ \
#                                                        --do_train \
#                                                        --do_eval \
#                                                        --mlm_loss \
#                                                        --col_loss \
#                                                        --train_batch_size 24 \
#                                                        --max_seq_length 512 \
#                                                        --num_train_epochs 30 \