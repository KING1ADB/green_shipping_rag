from flask import Flask, request, jsonify
from transformers import RagTokenizer, RagTokenForGeneration
import torch

app = Flask(__name__)

# Load model and tokenizer
tokenizer = RagTokenizer.from_pretrained('./fine_tuned_model')
model = RagTokenForGeneration.from_pretrained('./fine_tuned_model')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    inputs = tokenizer(query, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify(result=result)

if __name__ == "__main__":
    app.run(debug=True)