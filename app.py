from flask import Flask, request, jsonify, render_template
from chatbot import Chatbot
from datetime import datetime
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

app = Flask(__name__)

print("Loading pretrained language models...")
phi2_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
phi2_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", 
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically handles GPU/CPU allocation
    load_in_4bit=True  # 4-bit quantization
)

if torch.cuda.is_available():
    phi2_model = phi2_model.to('cuda')

bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

chatbot = Chatbot(model_path='chatbot_model.h5', 
                 phi2_model=phi2_model, phi2_tokenizer=phi2_tokenizer,
                 bert_model=bert_model, bert_tokenizer=bert_tokenizer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.get_json()
    message = data.get('message', '')
    user_id = data.get('user_id', 'default')
    response = chatbot.chat(message, user_id)
    return jsonify({'response': response})

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    message = data.get('message', '')
    response = data.get('response', '')
    is_correct = data.get('is_correct', False)
    user_id = data.get('user_id', 'anonymous')
    
    if not is_correct:
        feedback_entry = {
            'message': message,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id
        }
        
        try:
            with open('feedback.json', 'a') as f:
                f.write(json.dumps(feedback_entry) + '\n')
        except Exception as e:
            print(f"Error saving feedback: {e}")
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    print("Starting Flask server with enhanced chatbot...")
    app.run(host='0.0.0.0', port=5000, debug=True)