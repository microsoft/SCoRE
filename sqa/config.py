from typing import List
from dataclasses import dataclass, field

@dataclass
class Config:
    # data dir
    reader_pkl: str = ""

    # sketch file
    sketch_pkl: str = ""
    sketch_test_pkl: str = ""
    sketch_action_file: str = ""
    prod_file: str = ""

    # config for roberta
    use_roberta: bool = False
    use_tablebert: bool = False
    roberta_path: bool = None
    roberta_lr: float = 1e-5
    roberta_finetune: bool = False

    # config for programmer
    token_embed_size: int = 300
    var_token_size: int = 256  # not used if use_roberta
    token_dropout: float = 0.5
    token_rnn_size: int = 256
    token_indicator_size: int = 16
    pos_embed_size: int = 64
    slot_dropout: float = 0.25
    prod_embed_size: int = 512
    prod_rnn_size: int = 512
    prod_dropout: float = 0.25
    op_embed_size: int = 128

    column_type_embed_size: int = 16
    column_indicator_size: int = 16
    slot_hidden_score_size: int = 512

    model_type: str = "struct"

    # config for optimization
    lr: float = 5e-4
    l2: float = 1e-5
    clip_norm: float = 5

    # config for DistributedDataParallel
    nodes: int = 0
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    gpus: int = 1
    world_size: int = 1
    node_rank: int = 0
    addr: str = "127.0.0.1"
    port: str = "8003"

    # config for training
    epochs: int = 10
    batch_size: int = 1
    seed: int = 3264

    model_path: str = ""


configs = {}

configs["1_gpu_vanilla"] = Config()

configs["1_gpu_score"] = Config(
    token_dropout=0.1,

    use_roberta=True,
    use_tablebert=True,
    roberta_path=None,
    roberta_lr=1e-5,
    roberta_finetune=True,

    token_embed_size=1024,
)

configs["2_gpu_score"] = Config(
    token_dropout=0.1,

    use_roberta=True,
    use_tablebert=True,
    roberta_path=None,
    roberta_lr=1e-5,
    roberta_finetune=True,

    token_embed_size=1024,

    nodes=1,
    gpu_ids=[0, 1],
    gpus=2,
    world_size=2,
    node_rank=0,
    port="8006",

    batch_size=4,
)