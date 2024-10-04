# green_shipping_rag
This project fine-tunes a RAG model using OpenAI's API and integrates Named Entity Recognition (NER) functionality.

# Step 1: Set up environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Step 2: Clone repository
git clone https://github.com/KING1ADB/green_shipping_rag.git

cd green_shipping_rag

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Set OpenAI API Key
export OPENAI_API_KEY='your_openai_api_key'  # or `set` on Windows

# Steps 5-10: Run scripts
python src/data_collection.py
python src/data_annotation.py
python src/data_preprocessing.py
python src/model_training.py
python src/model_evaluation.py
python src/app.py

# Step 11: Open browser and search
http://127.0.0.1:5000/search?query=YOUR_QUERY