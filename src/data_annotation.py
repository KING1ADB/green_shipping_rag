import json
import openai
import os

# Load your OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

def annotate_data():
    annotated_data = []
    with open('data/raw_dataset.jsonl', 'r') as f:
        for line in f:
            paper = json.loads(line)
            # Use OpenAI's NER to annotate entities
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Identify entities in the following text: {paper['text']}."}]
            )
            paper['entities'] = response['choices'][0]['message']['content']
            annotated_data.append(paper)

    with open('data/annotated_dataset.jsonl', 'w') as f:
        for paper in annotated_data:
            f.write(json.dumps(paper) + '\n')

if __name__ == "__main__":
    annotate_data()