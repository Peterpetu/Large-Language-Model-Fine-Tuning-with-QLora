import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
from torch import nn
from .callbacks import PrintLossCallback
from .utils import print_trainable_parameters, print_weight_range, print_quantized_weights

# Set the seed for reproducibility
torch.manual_seed(42)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the custom tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/gpt3-finnish-small")
    model = AutoModelForCausalLM.from_pretrained("TurkuNLP/gpt3-finnish-small").to(device)

    # Prepare model for 4-bit training and freeze parameters
    model = prepare_model_for_kbit_training(model)
    freeze_model_parameters(model)

    # Q/Lora specific configuration
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query_key_value"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    # Print the number of trainable parameters
    print_trainable_parameters(model)

    # Load and preprocess the dataset
    train_dataset, eval_dataset = load_and_preprocess_dataset(tokenizer)

    # Define and execute training
    execute_training(model, tokenizer, train_dataset, eval_dataset)

    # Print the range of the model weights and quantized weights
    print_weight_range(model)
    print_quantized_weights(model)

    # Save and evaluate the model
    save_and_evaluate_model(model, tokenizer)

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

def load_and_preprocess_dataset(tokenizer):
    data_files = {"train": "data/corrected_finnish_dataset.json"}
    raw_datasets = load_dataset("json", data_files=data_files)
    tokenized_datasets = raw_datasets.map(lambda x: format_for_training(x, tokenizer), batched=True)
    
    # Splitting the dataset into training and evaluation sets
    train_dataset = tokenized_datasets['train'].train_test_split(test_size=0.1)['train']
    eval_dataset = tokenized_datasets['train'].train_test_split(test_size=0.1)['test']
    
    return train_dataset, eval_dataset

def format_for_training(examples, tokenizer):
    formatted_input = [f'instruction: {instr} context: {ctx}' for instr, ctx in zip(examples['instruction'], examples['context'])]
    inputs = tokenizer(formatted_input, truncation=True, padding='max_length', return_tensors="pt")
    inputs["labels"] = tokenizer(examples['response'], truncation=True, padding='max_length', return_tensors="pt")['input_ids']
    
    # Verify that the inputs and labels have the same length
    assert len(inputs["input_ids"]) == len(inputs["labels"]), "Inputs and labels length mismatch"
    
    # Print the first few examples to verify they are correct
    print("First few examples:")
    for i in range(min(2, len(inputs["input_ids"]))):
        print(f"Input: {tokenizer.decode(inputs['input_ids'][i])}")
        print(f"Label: {tokenizer.decode(inputs['labels'][i])}")
    
    return inputs

def execute_training(model, tokenizer, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir="./training_outputs",
    num_train_epochs=0.1,  # Adjusted based on multi-epoch considerations
    per_device_train_batch_size=1,  # Adjusted batch size
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=3,  # Adjusted batch size
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[PrintLossCallback()]
    )
    trainer.train()

def save_and_evaluate_model(model, tokenizer):
    model.save_pretrained("../test_model2")
    trainer = Trainer(model=model, tokenizer=tokenizer)
    eval_results = trainer.evaluate()

    # Save evaluation results to a file
    with open('./training_outputs/eval_results.json', 'w') as f:
        json.dump(eval_results, f)

    # Print evaluation results
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    main()
