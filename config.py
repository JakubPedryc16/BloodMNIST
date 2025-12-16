from dataclasses import dataclass

@dataclass
class Config:
    data_flag: str = "bloodmnist"

    batch_size: int = 128

    num_workers: int = 4

    n_epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"

    output_dir: str = "outputs"

    experiment_name: str = "exp_baseline_cnn"

    model_type: str = "simple_cnn"

    use_augment: bool = True

    seed: int = 42
