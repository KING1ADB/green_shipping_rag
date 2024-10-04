import requests
import json

def collect_data():
    # Placeholder for actual data source
    endpoint = "https://api.example.com/research-papers"
    response = requests.get(endpoint)
    
    if response.status_code == 200:
        papers = response.json()
        with open('data/raw_dataset.jsonl', 'w') as f:
            for paper in papers:
                f.write(json.dumps(paper) + '\n')
    else:
        print("Failed to collect data")

if __name__ == "__main__":
    collect_data()