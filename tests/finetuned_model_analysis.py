from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

def load_models():
    print("Loading original and finetuned models...")
    original_model = AutoModelForCausalLM.from_pretrained("TurkuNLP/gpt3-finnish-small")

    # Load finetuned model
    finetuned_model_path = '../merged_model'  # Replace with the correct path if different
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)

    # Load base model and merge LoRa adapter
    base_model = AutoModelForCausalLM.from_pretrained("TurkuNLP/gpt3-finnish-small")
    peft_model = PeftModel.from_pretrained(base_model, model_id='../test_model2')
    merged_model = peft_model.merge_and_unload()
    print("Models loaded and merged model created.")

    return original_model, finetuned_model, merged_model

def compare_model_weights(original_model, finetuned_model):
    print("\nComparing model weights...")
    for (orig_name, orig_param), (finetuned_name, finetuned_param) in zip(original_model.named_parameters(), finetuned_model.named_parameters()):
        if orig_name == finetuned_name:
            weight_difference = torch.norm(orig_param.data - finetuned_param.data).item()
            print(f"Layer: {orig_name}, Weight Change (L2 Norm): {weight_difference}")

def check_quantization(model):
    print("\nChecking for quantization in the model...")
    for name, param in model.named_parameters():
        data_type = param.dtype
        mean = param.mean().item()
        std = param.std().item()
        unique_values = torch.unique(param)
        unique_values_count = len(unique_values)
        quantization_possible = "Yes" if unique_values_count < 100 else "No"
        print(f"{name}: Data Type: {data_type}, Mean: {mean}, Std: {std}, Unique Values: {unique_values_count}, Quantization Possible: {quantization_possible}")

# Main execution
original_model, finetuned_model, merged_model = load_models()
compare_model_weights(original_model, finetuned_model)
check_quantization(merged_model)
