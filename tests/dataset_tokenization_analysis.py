import json
from collections import Counter
from transformers import PreTrainedTokenizerFast

# Load tokenizer configurations
with open('tokenizer_config.json', 'r', encoding='utf-8') as config_file:
    tokenizer_config = json.load(config_file)

with open('special_tokens_map.json', 'r', encoding='utf-8') as tokens_map_file:
    special_tokens_map = json.load(tokens_map_file)

# Load the custom tokenizer with configurations
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json", **tokenizer_config)

# Update tokenizer with special tokens from the map
tokenizer.add_special_tokens(special_tokens_map)

# Ensure pad token is set correctly
if tokenizer.pad_token is None:
    raise ValueError("Tokenizer does not have a defined padding token.")

# Print special tokens and their IDs
print("Special Tokens Map:", tokenizer.special_tokens_map)
print("Special Tokens IDs:", tokenizer.all_special_ids)

# Load the entire dataset
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Tokenize the entire dataset and perform analysis
def analyze_dataset(data):
    token_counts = Counter()
    sentence_lengths = []
    special_token_usage = Counter()
    oov_tokens = Counter()

    for entry in data:
        encoded = tokenizer.encode(entry['instruction'] + ' ' + entry['context'], add_special_tokens=True)
        token_counts.update(encoded)
        sentence_lengths.append(len(encoded))
        for token_id in encoded:
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token in special_tokens_map.values():
                special_token_usage[token] += 1
            if token_id == tokenizer.unk_token_id:
                oov_tokens[token] += 1

    return token_counts, sentence_lengths, special_token_usage, oov_tokens

dataset = load_dataset('../data/corrected_finnish_dataset.json')
token_counts, sentence_lengths, special_token_usage, oov_tokens = analyze_dataset(dataset)

# Print analysis results for the entire dataset
print("\n--- Dataset Analysis ---")
most_common_tokens = [(tokenizer.convert_ids_to_tokens(token_id), count) for token_id, count in token_counts.most_common(10)]
print("Most Common Tokens:", most_common_tokens)
print(f"Sentence Lengths: Avg = {sum(sentence_lengths) / len(sentence_lengths):.2f}, Min = {min(sentence_lengths)}, Max = {max(sentence_lengths)}, Median = {sorted(sentence_lengths)[len(sentence_lengths) // 2]} tokens")
print("Special Token Usage:", special_token_usage if special_token_usage else "No Special Tokens used.")
print("Out-of-Vocabulary Tokens:", oov_tokens if oov_tokens else "No Out-of-Vocabulary Tokens.")

# Load and tokenize a sample of the dataset for demonstration
def load_sample_dataset(file_path, num_samples=5):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data[:num_samples]

sample_data = load_sample_dataset('../data/corrected_finnish_dataset.json')

# Demonstrate tokenization on sample data
for entry in sample_data:
    input_text = f'instruction: {entry["instruction"]} context: {entry["context"]}'
    encoded_sample = tokenizer.encode(input_text, add_special_tokens=True)
    decoded_sample = tokenizer.decode(encoded_sample)

    print("\nSample Text:", input_text)
    print("Encoded Length:", len(encoded_sample))
    print("Encoded:", encoded_sample)
    print("Decoded:", decoded_sample)