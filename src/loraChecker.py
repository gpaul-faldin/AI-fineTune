import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from typing import List, Dict
import json
import time
import os

class ModelTester:
    def __init__(self, base_model_name: str, lora_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self._print_model_layers(self.base_model)

        if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            raise ValueError(f"No adapter_config.json found in {lora_path}")

        with open(os.path.join(lora_path, "adapter_config.json"), 'r') as f:
            config_dict = json.load(f)

        print("\nLoaded LoRA config:", config_dict)

        lora_config = LoraConfig(
            r=config_dict.get("r", 32),
            lora_alpha=config_dict.get("lora_alpha", 16),
            target_modules=config_dict.get("target_modules", ["Wqkv", "fc1", "fc2"]),
            lora_dropout=config_dict.get("lora_dropout", 0.05),
            bias=config_dict.get("bias", "none"),
            task_type="CAUSAL_LM"
        )

        print("Loading LoRA model...")
        self.lora_model = PeftModel.from_pretrained(
            self.base_model,
            lora_path,
            config=lora_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self._verify_lora_loading(self.lora_model)
        print("Models loaded successfully!")

    def _print_model_layers(self, model):
        """Print model layer names to verify architecture."""
        print("\nModel layer names:")
        for name, _ in model.named_modules():
            if any(target in name for target in ["Wqkv", "fc1", "fc2", "out_proj"]):
                print(f"Found target layer: {name}")

    def _verify_lora_loading(self, model):
        """Verify that LoRA parameters are properly loaded."""
        print("\nVerifying LoRA parameters:")
        lora_params_found = False
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                lora_params_found = True
                print(f"\nLoRA parameter: {name}")
                print(f"Shape: {param.shape}")
                print(f"Mean: {param.mean().item():.6f}")
                print(f"Std: {param.std().item():.6f}")
                print(f"Max: {param.max().item():.6f}")
                print(f"Min: {param.min().item():.6f}")
        
        if not lora_params_found:
            print("WARNING: No LoRA parameters found in the model!")

    def get_response(self, model, prompt: str, temperature: float = 0.1) -> str:
        """Generate a response from the specified model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        model.eval()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,  # Added top_k for better diversity
                num_return_sequences=1,
                no_repeat_ngram_size=3  # Prevent repetitive text
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def compare_responses(self, prompts: List[Dict], temperatures: List[float] = [0.1, 0.7]) -> Dict:
        """Compare responses between base and LoRA models at different temperatures."""
        results = []

        for prompt_dict in prompts:
            prompt = prompt_dict["prompt"]
            expected = prompt_dict.get("expected", None)

            prompt_results = {
                "prompt": prompt,
                "expected": expected,
                "comparisons": []
            }

            for temp in temperatures:
                # Set same seed for reproducibility
                torch.manual_seed(42)
                base_response = self.get_response(self.base_model, prompt, temp)
                torch.manual_seed(42)
                lora_response = self.get_response(self.lora_model, prompt, temp)
                prompt_results["comparisons"].append({
                    "temperature": temp,
                    "base_response": base_response,
                    "lora_response": lora_response
                })
            results.append(prompt_results)
        return results

def save_results(results: Dict, output_file: str):
    """Save test results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    # Model paths
    BASE_MODEL = "microsoft/phi-2"
    LORA_PATH = "./model/final"  # Adjust to your LoRA path

    tester = ModelTester(BASE_MODEL, LORA_PATH)

    test_prompts = [
        {
            "prompt": "What is Modern and Humanist France?",
            "expected": "A movement within the Union for a Popular Movement (UMP) created in August 2012"
        },
        {
            "prompt": "Who are the leaders of Modern and Humanist France?",
            "expected": "Jean-Pierre Raffarin, Luc Chatel, Jean Leonetti, and Marc Laffineur"
        },
        {
            "prompt": "Tell me about the Beauceron dog breed.",
            "expected": "A herding dog breed from Central France, also known as Berger de Beauce"
        }
    ]

    print("\nRunning comparison tests...")
    results = tester.compare_responses(test_prompts, temperatures=[0.1, 0.7])

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f"lora_test_results_{timestamp}.json"
    save_results(results, output_file)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()