from dataclasses import dataclass
from typing import List

#48gb of vram
@dataclass
class TrainingConfigA6000:
    model_name: str = "microsoft/phi-2"
    output_dir: str = "./outputs"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-5
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    max_length: int = 256
    bf16: bool = False
    fp16: bool = False
    use_cuda: bool = True
    cuda_device: str = "cuda:0"

#24gb of vram
@dataclass
class TrainingConfig3090:
    model_name: str = "microsoft/phi-2"
    output_dir: str = "./outputs"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-5
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    max_length: int = 256
    bf16: bool = False
    fp16: bool = True
    use_cuda: bool = True
    cuda_device: str = "cuda:0"

#8bg of vram
@dataclass
class TrainingConfig2080:
    model_name: str = "microsoft/phi-2"
    output_dir: str = "./outputs"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 6
    learning_rate: float = 3e-5
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    max_length: int = 256
    bf16: bool = False
    fp16: bool = True
    use_cuda: bool = True
    cuda_device: str = "cuda:0"

#The Target_module should be tweak according to your model, this one matches the phi-2 model
@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules = [  
        "Wqkv",
        "out_proj",
        "fc1",
        "fc2",
]


@dataclass
class DataConfig:
    data_path: str = "./dataset/modern.json"
    train_test_split: float = 0.1
