from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load the base model and the PeftModel with the adapter
base_model = AutoModelForCausalLM.from_pretrained("TurkuNLP/gpt3-finnish-small")
peft_model = PeftModel.from_pretrained(base_model, model_id='../test_model2')

# Merge the LoRa adapter into the base model
merged_model = peft_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained('../merged_model')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('./')

# Define a list of prompts for assessment
prompts = [
    "Moikka, mitä kuuluu?",
    "Kerro vitsi.",
    "Mikä on elämän tarkoitus?",
    "Kuvaile ilman sinistä väriä.",
    "Miten ilmastonmuutos vaikuttaa tulevaisuuteen?",
]

# Set the value of k for top-k predictions
k = 5

# Assess model responses for each prompt
for prompt in prompts:
    print("\nPrompt:", prompt)

    # Encode the prompt and generate a response using the base model
    inputs = tokenizer(prompt, return_tensors="pt")
    output_base = base_model.generate(inputs['input_ids'], max_length=50, do_sample=True, temperature=0.2, repetition_penalty=1.2, top_k=k)
    generated_text_base = tokenizer.decode(output_base[0], skip_special_tokens=True)
    print("Base model output:", generated_text_base)

    # Generate a response using the merged model
    output_merged = merged_model.generate(inputs['input_ids'], max_length=50, do_sample=True, temperature=0.2, repetition_penalty=1.2, top_k=k)
    generated_text_merged = tokenizer.decode(output_merged[0], skip_special_tokens=True)
    print("Merged model output:", generated_text_merged)

    # Get the top-k predicted tokens
    base_logits = output_base[0].squeeze()
    merged_logits = output_merged[0].squeeze()

    # Convert logits to float
    base_logits = base_logits.float()
    merged_logits = merged_logits.float()

    base_probs = torch.softmax(base_logits, dim=-1)
    merged_probs = torch.softmax(merged_logits, dim=-1)

    base_sorted_indices = torch.argsort(base_probs, descending=True)[:k]
    merged_sorted_indices = torch.argsort(merged_probs, descending=True)[:k]

    base_top_tokens = tokenizer.convert_ids_to_tokens(base_sorted_indices.tolist())
    merged_top_tokens = tokenizer.convert_ids_to_tokens(merged_sorted_indices.tolist())

    print(f"Top-{k} Predicted Tokens (Base Model):", base_top_tokens)
    print(f"Top-{k} Predicted Tokens (Merged Model):", merged_top_tokens)
