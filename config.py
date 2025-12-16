# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # dataset 
    data_flag: str = "bloodmnist"

    # batch size for training
    batch_size: int = 128

    # number of workers for DataLoader
    num_workers: int = 4

    # training settings
    n_epochs: int = 25        
    lr: float = 1e-3          # learning rate
    weight_decay: float = 1e-4  # L2 regularization
    optimizer: str = "adam"   

    # output 
    output_dir: str = "outputs"

    experiment_name: str = "exp_baseline_cnn"

    model_type: str = "simple_cnn"  

    use_augment: bool = True

    seed: int = 42
