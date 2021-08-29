# Pre-Training SCoRe Using Synthesized Data

This directory contains code for further pre-training RoBERTa or BERT on synthesized data to generate SCoRe used for MWoZ or other tasks (SParC, CoSQL, and SQA).

## Download Synthesized Data

Download synthesized data files from
[here](https://drive.google.com/file/d/1L9fWYzwcsLujzT6EYU2jckdl9LaQKa_T/view?usp=sharing), unzip
it, and put `data/` under `score/` dir. The data directory structure looks as follows:
```
score/
 - data/
  - augment_all_context.txt
  - augment_all_mlm_context.txt
  - ...
```

## Pre-Training SCoRe

First, create a dir to save generated SCoRe checkpoints by running `mkdir score_logs_checkpoints` (under `score/` dir).

Uncomment corresponding running commands in `run_example.sh`, and run
```
./run_example.sh
```
for training different SCoRe checkpoints.

You should find saved SCoRe checkpoints in `score_logs_checkpoints/` dirs.
