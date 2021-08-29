# SCoRe: Pre-Training for Context Representation in Conversational Semantic Parsing

This repository contains code for the ICLR 2021 paper ["SCoRE: Pre-Training for Context Representation in Conversational Semantic Parsing"](https://openreview.net/forum?id=oyZxhRI2RiE).

If you use SCoRE in your work, please cite it as follows:

```
@inproceedings{yu2021SCoRE,
  title={{SCoRE}: Pre-Training for Context Representation in Conversational Semantic Parsing},
  author={Tao Yu and Rui Zhang and Oleksandr Polozov and Christopher Meek and Ahmed Hassan Awadallah},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=oyZxhRI2RiE}
}
```

## Environment Setup

At the time of development, we used the same environment setup as [RAT-SQL](https://github.com/microsoft/rat-sql).
It assumes Python 3.7+ and CUDA 10.1.
Thus, the simplest environment setup for all the experiments except SQA (find SQA's environment setup in `sqa/README.md`) is:

``` bash
docker pull pytorch/pytorch:1.5-cuda10.1-cudnn7-devel
docker tag pytorch/pytorch:1.5-cuda10.1-cudnn7-devel score
docker run -it -v /path/to/this/repo:/workspace score
# or using GPUs
docker run --gpus 2 -it -v /path/to/this/repo:/workspace score
```

## Run Experiments

Code and running commands for running all the experiments can be found in the following dirs.
First, synthesize (or download) pre-training data and train a SCoRE checkpoint:
- `data_synthesis`: Synthesize Contextual Pre-Training Data
- `SCoRE`: Pre-Training SCoRE Using Synthesized Data

Then, to use the trained checkpoint as a base language model for conversational semantic parsing tasks:
- `mwoz`: SCoRE for Dialog State Tracking (MWoZ)
- `sqa`: SCoRE for Sequential Question Answering (SQA)
- `sparc_cosql`: SCoRE for Context-Dependent Semantic Parsing (SParC and CoSQL)


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of
Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion
or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those
third-party's policies.
