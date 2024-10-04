from transformers import RagTokenizer, RagTokenForGeneration
import torch
import json

def load_validation_data():
    with open('data/validation_dataset.jsonl', 'r') as f:
        return [json.loads(line) for line in f]

def evaluate_model():
    tokenizer = RagTokenizer.from_pretrained('./fine_tuned_model')
    model = RagTokenForGeneration.from_pretrained('./fine_tuned_model')

    validation_data = load_validation_data()

    for paper in validation_data:
        inputs = tokenizer(paper['text'], return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(**inputs)
        
        # Decode and print results
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    evaluate_model()