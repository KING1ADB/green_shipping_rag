import openai
import json

# Load your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

# Step 1: Prepare Training Data
def prepare_training_data():
    training_data = [
        {"prompt": "What are the benefits of green shipping?", "completion": "Green shipping reduces emissions, improves fuel efficiency, and enhances corporate image."},
        {"prompt": "What is a green port?", "completion": "A green port minimizes its environmental impact by implementing sustainable practices."},
    ]
    
    # Save to a JSONL file
    with open('data/training_data.jsonl', 'w') as f:
        for entry in training_data:
            f.write(json.dumps(entry) + '\n')

# Step 2: Fine-Tune the RAG Model
def fine_tune_model():
    response = openai.FineTune.create(
        training_file="data/training_data.jsonl",
        model="gpt-3.5-turbo"
    )
    print("Fine-tuning job ID:", response['id'])

if __name__ == "__main__":
    prepare_training_data()
    fine_tune_model()