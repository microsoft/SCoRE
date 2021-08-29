# SCoRE for Sequential Question Answering (SQA)

This is the PyTorch code for the SQA experiment in our paper.


## Dependency
We recommend Anaconda and please run the following to setup environment.
```
conda create -n score python=3.7
conda activate score
pip install -r requirements.txt
```
Furthermore, Java and [CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) with version [stanford-corenlp-4.1.0](http://nlp.stanford.edu/software/stanford-corenlp-4.1.0.zip) are needed for data preprocessing.

You also need to download the [GloVe](http://nlp.stanford.edu/data/glove.42B.300d.zip) word embeddings at `data/glove/glove.42B.300d.txt`.

## Data Download and Preprocessing
We have downloaded the [SQA](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54253) dataset at `data/SQA/SQA_release_1.0`. We have also corrected some annotation errors.
Furthermore, since SQA is based on WikiTableQuestions, we also downloaded tables at `data/SQA/SQA_release_1.0/tables.jsonl` and `data/SQA/raw_input` for preprocessing purpose.

To run data preprocessing, simply run the following:
```
./preprocess_sqa.sh
```

This preprocessing takes several steps, and the step of searching programs can take ~10 hours on SQA.
To save time, we have saved the preprocessed data at [here](https://drive.google.com/file/d/1-qGIMwpKKmndBMKvBK6Y9zNyFTsplyxc/view?usp=sharing), and you can put it as `data/processed_sqa`.

## Model Training on SQA
First, download the pretrained SCoRE model at [here](https://drive.google.com/file/d/1NkqQo095c1h99gi5BXB0xlf4o1JLHEv4/view?usp=sharing), and put it as `models/augment_all_context_roberta_ep3`.

To train the model on SQA, run
```
./train.sh
```
This will produce saved checkpoints in `checkpoints/` and logs in `log/`.

## Model Evaluation on SQA

You can download our saved checkpoints at [here](https://drive.google.com/file/d/1ud9e70iT3jdOUHTTLuOeMjZHhtxOjUSv/view?usp=sharing), and put it as `checkpoints/train_eval_score_devacc0.624.pt`.
To perform evaluation, run
```
./eval.sh
```
This will load a saved checkpoint, evaluate it on the SQA dev and test sets, and print the result:

Dev:
```
ALL SEQ Q1 Q2 Q3
62.4 35.6 74.5 61.4 51.9
```
Test:
```
ALL SEQ Q1 Q2 Q3
65.4 38.5 78.4 65.3 55.1
```

## Acknowledgement
The implementation is based on the following papers:
- [Learning Semantic Parsers from Denotations with Latent Structured Alignments and Abstract Programs](https://github.com/berlino/weaksp_em19), Wang et al., EMNLP 2019.
- [Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision](https://github.com/crazydonkey200/neural-symbolic-machines), Chen et al., ACL 2017.