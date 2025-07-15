import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import random
import json
import spacy
from datetime import datetime
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
import pickle
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class Chatbot:
    
    def __init__(self, model_path=None, phi2_model=None, phi2_tokenizer=None, 
                 bert_model=None, bert_tokenizer=None):
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_sm")
        self.context = {}
        self.stop_words = set(stopwords.words('english'))
        self.memory = defaultdict(dict)
        self.conversation_history = defaultdict(list)
        
        # Initialize pretrained models
        self.phi2_model = phi2_model
        self.phi2_tokenizer = phi2_tokenizer
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        
        self.initialize_resources()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = self.create_model()
            self.response_model = self.create_response_model()
    
    def initialize_resources(self):
        """Initialize all required resources including intents and knowledge base."""
        with open('intents.json') as file:
            self.intents = json.load(file)
        
        self.knowledge_base = {
            'qa_pairs': [
                {
                    'question': 'What is Python?',
                    'answer': 'Python is a high-level, interpreted programming language known for its readability and versatility.',
                    'keywords': ['python', 'programming', 'language']
                },
                {
                    'question': 'How to install Python?',
                    'answer': 'You can download Python from the official website python.org and follow the installation instructions for your operating system.',
                    'keywords': ['install', 'download', 'python']
                },
                {
                    'question': 'What is machine learning?',
                    'answer': 'Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.',
                    'keywords': ['machine learning', 'ai', 'artificial intelligence']
                }
            ],
            'facts': [
                'The first computer virus was created in 1983.',
                'The Python programming language was named after Monty Python, not the snake.',
                'There are approximately 700 programming languages in existence today.'
            ],
            'jokes': [
                'Why do programmers prefer dark mode? Because light attracts bugs!',
                'How many programmers does it take to change a light bulb? None, that\'s a hardware problem!',
                'Why do Java developers wear glasses? Because they don\'t C#!'
            ]
        }
        
        for idx, qa in enumerate(self.knowledge_base['qa_pairs']):
            self.intents['intents'].append({
                'tag': f'knowledge_{idx}',
                'patterns': [qa['question']],
                'responses': [qa['answer']],
                'context': qa.get('keywords', [])
            })
        
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = ['?', '!', '.', ',']
        
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        self.words = [self.lemmatizer.lemmatize(word.lower()) 
                     for word in self.words if word not in self.ignore_letters]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        self.setup_response_generation()
        self.setup_similarity_analyzer()
    
    def setup_response_generation(self):
        """Initialize components for response generation."""
        all_text = []
        for intent in self.intents['intents']:
            all_text.extend(intent['patterns'])
            all_text.extend(intent['responses'])
        
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(all_text)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_sequence_len = max([len(x.split()) for x in all_text])
    
    # def setup_similarity_analyzer(self):
    #     """Initialize TF-IDF vectorizer for semantic similarity."""
    #     self.all_patterns = []
    #     for intent in self.intents['intents']:
    #         self.all_patterns.extend(intent['patterns'])
        
    #     self.tfidf_vectorizer = TfidfVectorizer(
    #         tokenizer=self.clean_up_sentence, 
    #         stop_words='english'
    #     )
    #     self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_patterns)

    def setup_similarity_analyzer(self):
        """Initialize TF-IDF vectorizer for semantic similarity."""
        self.all_patterns = []
        for intent in self.intents['intents']:
            self.all_patterns.extend(intent['patterns'])
        
        # Modified to include token_pattern and handle stop words properly
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self.clean_up_sentence,
            token_pattern=None,  # Explicitly set to None since we're using tokenizer
            stop_words=list(self.stop_words)  # Convert stop_words set to list
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_patterns)
    
    def create_model(self):
        """Create and compile the intent classification model."""
        model = Sequential()
        model.add(Dense(256, input_shape=(len(self.words),), activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(len(self.classes), activation='softmax'))
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    def create_response_model(self):
        """Create and compile the response generation model."""
        model = Sequential()
        model.add(Embedding(self.vocab_size, 128, input_length=1))
        model.add(LSTM(256))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.vocab_size, activation='softmax'))
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    def train_model(self):
        """Train both the intent classification and response generation models."""
        training = []
        output_empty = [0] * len(self.classes)
        
        for doc in self.documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
            
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        
        self.model = self.create_model()
        self.model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=8, verbose=1)
        
        sequences = self.tokenizer.texts_to_sequences(self.all_patterns)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_len, padding='post')
        
        X = padded_sequences[:, :-1]
        y = padded_sequences[:, 1:]
        
        y_flat = y.reshape(-1)
        y_categorical = np.zeros((len(y_flat), self.vocab_size))
        for i, word_index in enumerate(y_flat):
            if word_index > 0:
                y_categorical[i, word_index] = 1
        
        X_flat = X.reshape(-1, 1)
        
        self.response_model = self.create_response_model()
        self.response_model.fit(X_flat, y_categorical, epochs=100, batch_size=16, verbose=1)
    
    def save_model(self, model_path='chatbot_model.h5'):
        """Save the trained models and vocabulary."""
        self.model.save(model_path)
        self.response_model.save('response_' + model_path)
        
        with open('words.pkl', 'wb') as f:
            pickle.dump(self.words, f)
        with open('classes.pkl', 'wb') as f:
            pickle.dump(self.classes, f)
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        print(f"Models saved to {model_path} and response_{model_path}")
    
    def load_model(self, model_path):
        """Load pre-trained models and vocabulary."""
        self.model = load_model(model_path)
        self.response_model = load_model('response_' + model_path)
        
        with open('words.pkl', 'rb') as f:
            self.words = pickle.load(f)
        with open('classes.pkl', 'rb') as f:
            self.classes = pickle.load(f)
        with open('tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print(f"Models loaded from {model_path} and response_{model_path}")
    
    def clean_up_sentence(self, sentence):
        """Tokenize and lemmatize a sentence."""
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) 
                         for word in sentence_words 
                         if word.lower() not in self.stop_words and word not in self.ignore_letters]
        return sentence_words
    
    def bag_of_words(self, sentence):
        """Convert sentence to bag of words vector."""
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, word in enumerate(self.words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)
    
    def predict_class(self, sentence):
        """Predict the intent class of a sentence."""
        query_vec = self.tfidf_vectorizer.transform([sentence])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        most_similar_idx = similarities.argmax()
        
        if similarities[0, most_similar_idx] > 0.5:
            pattern = self.all_patterns[most_similar_idx]
            for i, intent in enumerate(self.intents['intents']):
                if pattern in intent['patterns']:
                    return [{'intent': intent['tag'], 'probability': str(similarities[0, most_similar_idx])}]

        if self.bert_model and self.bert_tokenizer:
            try:
                inputs = self.bert_tokenizer(sentence, return_tensors="tf", truncation=True, padding=True)
                outputs = self.bert_model(inputs)
                probs = tf.nn.softmax(outputs.logits, axis=-1)
                bert_pred = tf.argmax(probs, axis=-1).numpy()[0]
                
                if len(self.classes) > bert_pred:
                    return [{'intent': self.classes[bert_pred], 'probability': str(probs[0][bert_pred].numpy())}]
            except Exception as e:
                print(f"BERT prediction error: {e}")

        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        
        ERROR_THRESHOLD = 0.2
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [{'intent': self.classes[r[0]], 'probability': str(r[1])} for r in results]

    def generate_response_phi2(self, seed_text, max_length=150):
        """Generate response using Phi-2 model."""
        if not self.phi2_model or not self.phi2_tokenizer:
            return None
            
        prompt = f"""You are a helpful AI assistant. Provide a concise and accurate response to the following:
        
        User: {seed_text}
        Assistant:"""
        
        inputs = self.phi2_tokenizer(
            prompt, 
            return_tensors="pt", 
            return_attention_mask=False
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.phi2_model.generate(
                **inputs, 
                max_length=max_length,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.phi2_tokenizer.eos_token_id
            )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        generated_text = self.phi2_tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        generated_text = generated_text[len(prompt):].strip()
        generated_text = generated_text.split('\n')[0]
        generated_text = generated_text.split('.')[0] + '.' if '.' in generated_text else generated_text
        
        return generated_text.strip()
    
    def generate_response(self, seed_text, num_words=15):
        """Generate a response using either Phi-2 or the custom model."""
        if self.phi2_model and self.phi2_tokenizer:
            phi2_response = self.generate_response_phi2(seed_text)
            if phi2_response:
                return phi2_response
        
        try:
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
            
            predicted = self.response_model.predict(token_list, verbose=0)
            predicted_indices = np.argsort(predicted)[0][-num_words:]
            
            predicted_words = []
            for i in predicted_indices:
                for word, index in self.tokenizer.word_index.items():
                    if index == i:
                        predicted_words.append(word)
            
            response = ' '.join(predicted_words)
            response = re.sub(r'\s+([.,!?])', r'\1', response)
            response = response.capitalize()
            
            return response
        except Exception as e:
            print(f"Error in response generation: {e}")
            return None
    
    def extract_entities(self, text):
        """Extract named entities from text using spaCy."""
        doc = self.nlp(text)
        return {
            "persons": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
            "locations": [ent.text for ent in doc.ents if ent.label_ == "GPE"],
            "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
            "organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
            "nouns": [token.text for token in doc if token.pos_ == "NOUN"],
            "verbs": [token.lemma_ for token in doc if token.pos_ == "VERB"]
        }
    
    def get_response(self, intents_list, user_input=None, user_id="default"):
        """Get response for the highest probability intent."""
        if not intents_list:
            if user_input and self.phi2_model:
                return self.generate_response(user_input)
            return "I'm not sure how to respond to that. Could you rephrase?"

        tag = intents_list[0]['intent']
        probability = float(intents_list[0]['probability'])

        if probability < 0.5 or tag in ['jokes', 'facts']:
            if user_input and self.phi2_model:
                phi2_response = self.generate_response(user_input)
                if phi2_response:
                    return phi2_response

        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                base_response = random.choice(intent['responses'])
                
                if probability > 0.85 and user_input:
                    generated_response = self.generate_response(user_input)
                    if generated_response:
                        return f"{base_response} {generated_response}"
                
                if tag == 'jokes':
                    return f"{base_response} {random.choice(self.knowledge_base['jokes'])}"
                elif tag == 'facts':
                    return f"{base_response} {random.choice(self.knowledge_base['facts'])}"
                
                return base_response
        
        for qa_pair in self.knowledge_base['qa_pairs']:
            if fuzz.ratio(user_input.lower(), qa_pair['question'].lower()) > 65:
                return qa_pair['answer']
        
        if user_input and self.phi2_model:
            return self.generate_response(user_input)
        
        return "I'm not sure how to respond to that. Could you ask in a different way?"
    
    def handle_special_requests(self, message, entities, user_id="default"):
        """Handle special requests like facts, jokes, etc."""
        message_lower = message.lower()
        
        if 'remember' in message_lower and 'that' in message_lower:
            if self.memory[user_id]:
                return "I remember these things about you: " + "; ".join(
                    f"{k}: {v}" for k, v in self.memory[user_id].items())
            else:
                return "I don't seem to remember anything specific about you yet."
        
        if 'my name is' in message_lower:
            name = message.split('my name is')[-1].strip()
            self.memory[user_id]['name'] = name
            return f"Nice to meet you, {name}! I'll remember that."
        
        keywords = {
            'jokes': ['joke', 'funny', 'laugh', 'humor', 'hilarious'],
            'facts': ['fact', 'interesting', 'know', 'learn', 'information'],
            'help': ['help', 'assist', 'support', 'guide']
        }
        
        if any(word in message_lower for word in keywords['jokes']):
            return "Here's a joke for you: " + random.choice(self.knowledge_base['jokes'])
        
        if any(word in message_lower for word in keywords['facts']):
            return "Did you know? " + random.choice(self.knowledge_base['facts'])
        
        if 'what is' in message_lower or 'who is' in message_lower:
            nouns = entities.get('nouns', [])
            if nouns:
                return f"I have information about {nouns[0]}. Let me check my knowledge base."
        
        return None
    
    def chat(self, message, user_id="default", creative_mode=False):
        """Main chat method that handles the conversation flow."""
        if user_id not in self.context:
            self.context[user_id] = {
                'last_intent': None,
                'entities': {},
                'conversation_history': []
            }
        
        self.context[user_id]['conversation_history'].append(("user", message))
        self.conversation_history[user_id].append(message)
        
        entities = self.extract_entities(message)
        self.context[user_id]['entities'] = entities
        
        special_response = self.handle_special_requests(message, entities, user_id)
        if special_response:
            self.context[user_id]['conversation_history'].append(("bot", special_response))
            return special_response
        
        kb_response = self.check_knowledge_base(message)
        if kb_response:
            self.context[user_id]['conversation_history'].append(("bot", kb_response))
            return kb_response
        
        if (creative_mode or not kb_response) and self.phi2_model:
            phi2_response = self.generate_response(message)
            if phi2_response:
                self.context[user_id]['conversation_history'].append(("bot", phi2_response))
                return phi2_response
        
        ints = self.predict_class(message)
        response = self.get_response(ints, message, user_id)
        
        if not response:
            response = "I'm not sure how to respond to that. Could you try rephrasing your question?"
        
        self.context[user_id]['conversation_history'].append(("bot", response))
        if ints:
            self.context[user_id]['last_intent'] = ints[0]['intent']
        
        return response
    
    def check_knowledge_base(self, query):
        """Check if the query matches any knowledge base entry."""
        for qa_pair in self.knowledge_base['qa_pairs']:
            if fuzz.ratio(query.lower(), qa_pair['question'].lower()) > 70:
                return qa_pair['answer']
            
            for keyword in qa_pair.get('keywords', []):
                if keyword.lower() in query.lower():
                    return qa_pair['answer']
        
        if query.lower().startswith(('what is ', 'who is ', 'where is ', 'how to ')):
            if self.phi2_model:
                return self.generate_response(query)
        
        return None