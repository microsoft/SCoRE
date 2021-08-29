#! /bin/bash

ROOT=$(pwd)
echo $ROOT

# Prepare input files for CoreNLP
python preprocess_sqa.py --step=prepare_corenlp \
                         --sqa_path="data/SQA/SQA_release_1.0" \
                         --table_file="data/SQA/tables.jsonl" \
                         --corenlp_input_output_list="data/SQA/corenlp_input_output_list.txt" \
                         --corenlp_input_output="data/SQA/corenlp_input_output"

# Run CoreNLP over questions and answers
cd $ROOT/stanford-corenlp-4.1.0/
./run_corenlp.sh

# Prepare raw input for preprocessing using neural-symbolic-machines
cd $ROOT
python preprocess_sqa.py --step=prepare_raw_input \
                         --sqa_path="data/SQA/SQA_release_1.0" \
                         --corenlp_input_output="data/SQA/corenlp_input_output" \
                         --raw_input="data/SQA/raw_input"

# Run neural-symbolic-machines preprocessing
cd $ROOT/neural-symbolic-machines/
python preprocess_sqa_nsm.py \
       --raw_input_dir=$ROOT"/data/SQA/raw_input" \
       --processed_input_dir=$ROOT"/data/SQA/processed_input" \
       --max_n_tokens_for_num_prop=10 \
       --min_frac_for_ordered_prop=0.2 \
       --use_prop_match_count_feature \
       --expand_entities \
       --process_conjunction \
       --alsologtostderr

# Cache data into pkl files
cd $ROOT
python preprocess_sqa.py --step=cache_data \
                         --table_file="data/SQA/tables.jsonl" \
                         --train_file="data/SQA/processed_input/train_examples.jsonl" \
                         --dev_file="data/SQA/processed_input/dev_examples.jsonl" \
                         --test_file="data/SQA/processed_input/test_examples.jsonl" \
                         --embed_file="data/glove/glove.42B.300d.txt" \
                         --output_file="data/processed_sqa/sqa_glove_42B_minfreq_3.pkl"

# Search programs
python preprocess_sqa.py --step=search_program \
                         --exp_id="demomp" \
                         --max_sketch_length=9 \
                         --table_file="data/SQA/tables.jsonl" \
                         --train_file="data/SQA/processed_input/train_examples.jsonl" \
                         --dev_file="data/SQA/processed_input/dev_examples.jsonl" \
                         --test_file="data/SQA/processed_input/test_examples.jsonl"

# Cache Programs
python preprocess_sqa.py --step=cache_program \
                         --program_file_name="data/processed_sqa/demomp.train.programs" \
                         --section=train \
                         --output_filename="data/processed_sqa/train.pkl"

python preprocess_sqa.py --step=cache_program \
                         --program_file_name="data/processed_sqa/demomp.test.programs" \
                         --section=test \
                         --output_filename="data/processed_sqa/test.pkl"
