from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle
import os
from utils.data_preprocessing import DataPreprocessor
from models.seq2seq_model import Seq2SeqModel
from utils.logger import setup_logger

app = Flask(__name__)
logger = setup_logger('logs/app.log')

# Configuration
MODEL_TYPE = 't5'  # Options: 'lstm' or 't5'
MODEL_DIR = 'models/saved'
MAX_TEXT_LEN = 512
MAX_SUMMARY_LEN = 120  # Increased to ensure complete summaries

# Initialize models and tokenizers
preprocessor = None
lstm_model = None
encoder_model = None
decoder_model = None
text_tokenizer = None
summary_tokenizer = None
t5_model = None
t5_tokenizer = None

def load_lstm_model():
    """Loads LSTM model and tokenizers."""
    global preprocessor, lstm_model, encoder_model, decoder_model, text_tokenizer, summary_tokenizer
    try:
        logger.info("Loading LSTM model and tokenizers")
        preprocessor = DataPreprocessor(max_text_len=MAX_TEXT_LEN, max_summary_len=MAX_SUMMARY_LEN, glove_dim=50)
        glove_file = 'glove.6B.50d.txt'
        preprocessor.load_glove_embeddings(glove_file)
        
        with open(os.path.join(MODEL_DIR, 'text_tokenizer.pkl'), 'rb') as f:
            text_tokenizer = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'summary_tokenizer.pkl'), 'rb') as f:
            summary_tokenizer = pickle.load(f)
        
        embedding_matrix = preprocessor.create_embedding_matrix()
        lstm_model = Seq2SeqModel(
            x_voc=len(text_tokenizer.word_index) + 1,
            y_voc=len(summary_tokenizer.word_index) + 1,
            max_text_len=MAX_TEXT_LEN,
            max_summary_len=MAX_SUMMARY_LEN,
            glove_embedding_matrix=embedding_matrix,
            glove_dim=50,
            latent_dim=200
        )
        lstm_model.build_model()
        
        lstm_model.model = load_model(os.path.join(MODEL_DIR, 'seq2seq_model.h5'))
        encoder_model = load_model(os.path.join(MODEL_DIR, 'encoder_model.h5'))
        decoder_model = load_model(os.path.join(MODEL_DIR, 'decoder_model.h5'))
        
        lstm_model.encoder_model = encoder_model
        lstm_model.decoder_model = decoder_model
        logger.info("LSTM model and tokenizers loaded successfully")
    except Exception as e:
        logger.error("Failed to load LSTM model: %s", e)
        preprocessor = None
        lstm_model = None
        encoder_model = None
        decoder_model = None
        text_tokenizer = None
        summary_tokenizer = None
        raise

def load_t5_model():
    """Loads T5 model and tokenizer."""
    global t5_model, t5_tokenizer
    try:
        model_path = os.path.join(MODEL_DIR, 't5_model')
        tokenizer_path = os.path.join(MODEL_DIR, 't5_tokenizer')
        logger.info("Loading T5 model from %s and tokenizer from %s", model_path, tokenizer_path)
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"T5 model or tokenizer directory not found: {model_path}, {tokenizer_path}")
        t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
        t5_tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        logger.info("T5 model and tokenizer loaded successfully. Model config: %s", t5_model.config)
        logger.info("T5 model parameters: %d", sum(p.numel() for p in t5_model.parameters()))
    except Exception as e:
        logger.error("Failed to load T5 model: %s", e)
        t5_model = None
        t5_tokenizer = None
        raise

# Load the appropriate model based on MODEL_TYPE
try:
    if MODEL_TYPE == 'lstm':
        load_lstm_model()
    else:
        load_t5_model()
except Exception as e:
    logger.error("Model initialization failed: %s", e)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ''
    error = ''
    if request.method == 'POST':
        article_text = request.form.get('article_text', '').strip()
        if not article_text:
            error = 'Please enter article text.'
        else:
            try:
                if MODEL_TYPE == 'lstm' and preprocessor is not None and lstm_model is not None:
                    logger.info("Generating LSTM summary for text: %s", article_text[:50])
                    tokens = preprocessor.tokenize_english([article_text])[0]
                    seq = preprocessor.text_tokenizer.texts_to_sequences([' '.join(tokens)])[0]
                    input_seq = pad_sequences([seq], maxlen=MAX_TEXT_LEN, padding='post', truncating='post')
                    summary = lstm_model.generate_summary(input_seq, preprocessor.summary_tokenizer, MAX_SUMMARY_LEN)
                    logger.info("Generated LSTM summary: %s", summary)
                elif MODEL_TYPE == 't5' and t5_model is not None and t5_tokenizer is not None:
                    logger.info("Generating T5 summary for text: %s", article_text[:50])
                    input_text = f"summarize: {article_text}"
                    inputs = t5_tokenizer(
                        input_text,
                        max_length=MAX_TEXT_LEN,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    logger.info("Input token count: %d", inputs['input_ids'].shape[1])
                    outputs = t5_model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=MAX_SUMMARY_LEN,
                        min_length=20,
                        length_penalty=0.4,
                        num_beams=8,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                        do_sample=False
                    )
                    summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.info("Generated T5 summary: %s (token count: %d)", summary, len(t5_tokenizer.encode(summary)))
                    if not summary or summary.strip() == article_text[:len(summary)].strip():
                        logger.warning("Summary is empty or identical to input. Retrying with adjusted parameters.")
                        outputs = t5_model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=MAX_SUMMARY_LEN,
                            min_length=15,
                            length_penalty=0.3,
                            num_beams=10,
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                            do_sample=True,
                            top_p=0.9
                        )
                        summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        logger.info("Retry T5 summary: %s (token count: %d)", summary, len(t5_tokenizer.encode(summary)))
                else:
                    error = 'Model not properly initialized. Please check logs.'
            except Exception as e:
                logger.error("Error generating summary: %s", e)
                error = f'Error generating summary: {str(e)}'
    
    return render_template('index.html', summary=summary, error=error, article_text=request.form.get('article_text', ''))

if __name__ == '__main__':
    app.run(debug=True)