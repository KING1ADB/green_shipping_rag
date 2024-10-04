import json
from transformers import RagTokenForGeneration, RagTokenizer, Trainer, TrainingArguments

def load_data():
    with open('data/preprocessed_dataset.jsonl', 'r') as f:
        return [json.loads(line) for line in f]

def train_model():
    data = load_data()
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

    # Prepare dataset for Trainer
    train_dataset = [{"input_ids": entry['input_ids'], "labels": entry['entities']} for entry in data]

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained('./fine_tuned_model')

if __name__ == "__main__":
    train_model()