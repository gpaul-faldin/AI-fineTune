import json
from typing import Dict, List, Tuple
from datasets import Dataset
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_config, training_config, tokenizer):
        self.data_config = data_config
        self.training_config = training_config
        self.tokenizer = tokenizer

    def load_data(self) -> List[Dict]:
        with open(self.data_config.data_path, 'r') as f:
            data = json.load(f)

        processed_data = []
        for conv in data['conversations']:
            text = self.tokenizer.apply_chat_template(
                [{"role": m["role"], "content": m["content"]} for m in conv["messages"]],
                tokenize=False,
                add_generation_prompt=False
            )
            processed_data.append({'text': text})

        return processed_data

    def preprocess_function(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.training_config.max_length,
            padding="max_length",
            return_tensors=None,
        )

    def create_dataset(self) -> Tuple[Dataset, Dataset]:
        data = self.load_data()
        

        train_data, val_data = train_test_split(
            data, 
            test_size=0.05,  # 5% for validation
            random_state=42
        )
        

        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        train_tokenized = train_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training dataset",
        )

        val_tokenized = val_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation dataset",
        )

        return (
            train_tokenized.with_format("torch"),
            val_tokenized.with_format("torch")
        )