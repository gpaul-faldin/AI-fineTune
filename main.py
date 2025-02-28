from src.config import LoRAConfig, DataConfig, TrainingConfig2080, TrainingConfigA6000, TrainingConfig3090
from src.data_processing import DataProcessor
from src.model import Phi2Model
from src.train import train_model
import torch

import json
from typing import List, Dict

gpu = 'rtx2080'

def main():
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("No GPU available! Training will be slow.")
        return

    # Initialize configs
    
    if gpu == 'A6000':
        training_config = TrainingConfigA6000()
    elif gpu == '3090':
        training_config = TrainingConfig3090()
    else:
        training_config = TrainingConfig2080()
    
    # training_config = TrainingConfig2080()
    lora_config = LoRAConfig()
    data_config = DataConfig()

    
    # Setup model and tokenizer
    phi2 = Phi2Model(training_config, lora_config, '3090')
    tokenizer = Phi2Model.setup_tokenizer()
    model = phi2.setup_model()

    # Process data
    data_processor = DataProcessor(data_config, training_config, tokenizer)
    datasets = data_processor.create_dataset()

    # Train
    trainer = train_model(model, tokenizer, datasets, training_config)

    # Save final model
    trainer.save_model(training_config.output_dir + "/final")

if __name__ == "__main__":
    main()