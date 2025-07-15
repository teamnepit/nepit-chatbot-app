from chatbot import Chatbot
import tensorflow as tf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

def main():
    print("Initializing and training the enhanced chatbot with pretrained models...")
    
    print("Loading pretrained language models...")
    phi2_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    phi2_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", 
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if torch.cuda.is_available():
        phi2_model = phi2_model.to('cuda')

    bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    
    chatbot = Chatbot(phi2_model=phi2_model, phi2_tokenizer=phi2_tokenizer,
                     bert_model=bert_model, bert_tokenizer=bert_tokenizer)
    
    print("Training the neural network models...")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU for training...")
    
    chatbot.train_model()
    
    print("Saving trained models...")
    chatbot.save_model('chatbot_model.h5')
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()