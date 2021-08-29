## SCoRe for Dialog State Tracking (MWoZ)

This directory contains instructions to reproduce SCoRe experiments on [MWoZ](https://github.com/budzianowski/multiwoz). We employ [TripPy](https://arxiv.org/abs/2005.02877) as our base model for the task.

## TripPy Model Code Download

TripPy's up-to-date code can be found
[here](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public). We used an [old version of the code](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public/-/tree/b7896c8ed0a6506378353accfbd95c67a66e20ff).
Download and patch it to obtain our modified version:
```
git clone https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public.git trippy
cd trippy
git checkout b7896c8ed0a6506378353accfbd95c67a66e20ff
git apply -p0 ../mwoz/trippy.patch
```

You should obtain the following directory structure:
```
trippy/
 - run_dst.py
 - run_example.sh
 - utils_dst.py
 - dataset_multiwoz21.py
 - ...
```

## Data Download

Download data files from
[here](https://drive.google.com/file/d/13pF9V9DeO4wODWCre5Wk2dVwbTzgJjor/view?usp=sharing), unzip it
and put them under `trippy/data/` directory (run `mkdir data` if necessary). The directory structure looks as follows:
```
trippy/
  data/
   - max_len_512_rp
```

## Model Training and Evaluation on MWoZ

1) Create a directory to save logs and checkpoints:
```
mkdir trippy/trippy_logs_checkpoints
```

2) Change `BERTDIR` in `run_example.sh` to use [the SCoRe checkpoint for MWoZ](https://drive.google.com/file/d/1NwVEOMGBRmdBB-oFp--Pw-NhOeJ9yTO9/view?usp=sharing).

3) To train the TripPy model on MWoZ, run
```
./run_example.sh
```

You should find saved checkpoints and results in `trippy_logs_checkpoints/` dirs. All dev and test joint goal accuracies are printed at the end of `eval_dir_pred_all.log` file (the final test result is selected by the best dev result).


## Acknowledgement

The base model TripPy is introduced by [TripPy: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877).
