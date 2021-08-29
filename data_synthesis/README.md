## Synthesize Contextual Pre-Training Data

This directory contains code for synthesizing contextual pre-training data.

## Download Synthesized Data
You can directly download the generated synthetic data from [here](https://drive.google.com/file/d/1L9fWYzwcsLujzT6EYU2jckdl9LaQKa_T/view?usp=sharing).

`augment_multiwoz_512.txt` is used to pre-train SCoRe for MWoZ and `augment_all_context.txt` is for other tasks. Files with `mlm` in the filename also includes examples for pre-training with a MLM loss. For each example, numbers after `|||` usually represent the label for each column.


## Synthesizing Data (optional)
Run Jupyter Notebooks to synthesize pre-training data.
Download files from [here](https://drive.google.com/file/d/10bapWjlm4sTp8B2to6MiZeTO-nyQGFWk/view?usp=sharing), put all data files in `data/` dir.
