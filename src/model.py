import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

class Phi2Model:
    def __init__(self, training_config, lora_config, gpu):
        self.training_config = training_config
        self.lora_config = lora_config
        self.gpu = gpu

    def setup_model(self):
        torch.cuda.empty_cache()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.training_config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r = self.lora_config.r,
            lora_alpha = self.lora_config.lora_alpha,
            lora_dropout = self.lora_config.lora_dropout,
            bias = self.lora_config.bias,
            target_modules = self.lora_config.target_modules,
            task_type='CAUSAL_LM'
            )
        model = get_peft_model(model, lora_config)

        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        return model

    @staticmethod
    def setup_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer