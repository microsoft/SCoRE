# RAT-SQL for Conversational Semantic Parsing (SParC and CoSQL)

This directory contains instructions to reproduce SCoRe experiments on [SParC](https://yale-lily.github.io/sparc) and [CoSQL](https://yale-lily.github.io/cosql). We employ [RAT-SQL](https://github.com/microsoft/rat-sql) as our base model for the two tasks.

## RAT-SQL Model Code Download
RAT-SQL's up-to-date code can be found [here](https://github.com/microsoft/rat-sql). Download the
code and patch it to obtain our RoBERTa-based fork of RAT-SQL:

```
git clone https://github.com/microsoft/rat-sql ratsql
cd ratsql
git apply -p0 ../sparc_cosql/ratsql.patch
```

## Data Download
Download processed SParC and CoSQL data from
[here](https://drive.google.com/file/d/1_xBYGgi-mVCHd4dzqavigPzcEWzRL60r/view?usp=sharing), unzip it
and put them under `data/` dir (run `mkdir data`). The data directory structure looks as follows:

```
ratsql/
  data/
  - sparc_concat_q_contextbert_roberta
    - ...
  - cosql_concat_q_contextbert_roberta
    - ...
```

## Model Training and Evaluation
By default, this codebase uses RoBERTa as its base pretrained language model.
To use a [SCoRE checkpoint](https://drive.google.com/file/d/1eFkjeOXtc94tN21qfxDl8a9KQpjf3wcw/view?usp=sharing) instead and run experiments on SPaRC/CoSQL, find and edit 4 occurrences of `TODO(SCORE)`:
- `ratsql/models/spider/spider_enc.py`: path to pretrained SCoRE checkpoint; model file name
- `experiments/spider-bert-run.jsonnet`: path to SParC/CoSQL data; SCoRE checkpoint step for evaluation.

Then follow the instructions on [the official RAT-SQL README](https://github.com/microsoft/rat-sql#step-3-run-the-experiments) to train and evaluate a model on SParC and CoSQL.

## Acknowledgement

The base model RAT-SQL is introduced by ["RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers"](https://arxiv.org/abs/1911.04942).
