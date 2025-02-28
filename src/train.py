import math
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

def on_evaluate(self, args, state, control, **kwargs):
    metrics = state.log_history[-1]
    if "eval_loss" in metrics:
        perplexity = math.exp(metrics["eval_loss"])
        print(f"Perplexity: {perplexity:.2f}")

def train_model(model, tokenizer, datasets, config):
    train_dataset, eval_dataset = datasets

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="steps",
        save_steps=400,
        eval_steps=400,
        eval_strategy="steps",
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        # GPU specific settings
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        no_cuda=not config.use_cuda,
        report_to=[],
        logging_dir=None,
        logging_strategy="no",
        # Additional fp16 optimization settings
        fp16_opt_level="O2",          # Mixed precision training
        fp16_backend="auto",
        fp16_full_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if torch.cuda.is_available():
        print(f"GPU Memory before training: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    trainer.train()
    return trainer
