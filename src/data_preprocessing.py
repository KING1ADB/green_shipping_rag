from transformers import AutoTokenizer
import json

def preprocess_data():
    tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
    preprocessed_data = []

    with open('data/annotated_dataset.jsonl', 'r') as f:
        for line in f:
            paper = json.loads(line)
            inputs = tokenizer(paper['text'], truncation=True, padding='max_length', return_tensors='pt')
            preprocessed_data.append({"input_ids": inputs["input_ids"].tolist(), "entities": paper['entities']})

    with open('data/preprocessed_dataset.jsonl', 'w') as f:
        for entry in preprocessed_data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    preprocess_data()