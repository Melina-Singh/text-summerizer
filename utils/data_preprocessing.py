import numpy as np
import pandas as pd
import nltk
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger
import os

nltk.download('punkt', quiet=True)
logger = setup_logger('logs/app.log')

class DataPreprocessor:
    """Handles data loading, GloVe embedding preparation, and sequence processing for summarization."""
    
    def __init__(self, max_text_len=300, max_summary_len=50, glove_dim=50):
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.glove_dim = glove_dim
        self.text_tokenizer = Tokenizer(filters='')
        self.summary_tokenizer = Tokenizer(filters='')
        self.glove_embeddings = None
        logger.info("DataPreprocessor initialized with max_text_len=%d, max_summary_len=%d, glove_dim=%d", 
                    max_text_len, max_summary_len, glove_dim)
        
    def load_glove_embeddings(self, glove_file='glove.6B.50d.txt'):
        """Loads GloVe embeddings from file."""
        logger.info("Loading GloVe embeddings from %s", glove_file)
        embeddings = {}
        try:
            with open(glove_file, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    embeddings[word] = vector
            self.glove_embeddings = embeddings
            logger.info("GloVe embeddings loaded: %d words", len(embeddings))
        except Exception as e:
            logger.error("Failed to load GloVe embeddings: %s", e)
            raise ValueError(f"Failed to load GloVe embeddings: {e}")
        
    def create_embedding_matrix(self):
        """Creates embedding matrix for text tokenizer vocabulary."""
        if self.glove_embeddings is None:
            raise ValueError("GloVe embeddings not loaded")
        
        logger.info("Creating embedding matrix for text vocabulary")
        vocab_size = len(self.text_tokenizer.word_index) + 1
        embedding_matrix = np.zeros((vocab_size, self.glove_dim))
        oov_count = 0
        for word, idx in self.text_tokenizer.word_index.items():
            embedding_vector = self.glove_embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
            else:
                oov_count += 1
        logger.info("Embedding matrix created: vocab_size=%d, oov_words=%d", vocab_size, oov_count)
        return embedding_matrix
    
    def load_dataset(self, num_samples=2000):
        """Loads a mixed dataset (CNN/Daily Mail + Wikipedia) with alignment check."""
        wiki_samples = num_samples - 1000
        logger.info("Loading dataset with %d samples (1000 CNN/Daily Mail, %d Wikipedia)", num_samples, wiki_samples)
        try:
            # Load CNN/Daily Mail (1000 samples)
            cnn_dataset = load_dataset("cnn_dailymail", "3.0.0", split='train')
            cnn_df = pd.DataFrame({
                'text': cnn_dataset['article'][:1000],
                'summary': cnn_dataset['highlights'][:1000]
            })
            
            # Load Wikipedia (1000 samples)
            wiki_dataset = load_dataset("wikipedia", "20220301.en", split='train', trust_remote_code=True)
            wiki_texts = []
            wiki_summaries = []
            logger.debug("Wikipedia dataset structure: %s", str(wiki_dataset.column_names))
            
            wiki_count = 0
            i = 0
            while wiki_count < wiki_samples and i < len(wiki_dataset):
                article = wiki_dataset[i]
                if isinstance(article, dict) and 'text' in article:
                    text = article['text'].split('\n')[0][:1000]
                    summary_text = article['text']
                else:
                    logger.debug("Article %d is a string, treating as raw text", i)
                    text = str(article)[:1000]
                    summary_text = str(article)
                
                summary = ' '.join(nltk.sent_tokenize(summary_text)[:2])[:200]
                if text.strip() and summary.strip():
                    # Check alignment (basic keyword overlap)
                    text_tokens = set(nltk.word_tokenize(text.lower()))
                    summary_tokens = set(nltk.word_tokenize(summary.lower()))
                    overlap = len(text_tokens.intersection(summary_tokens)) / len(summary_tokens) if summary_tokens else 0
                    if overlap < 0.2:
                        logger.debug("Skipping article %d: low alignment (overlap=%.2f)", i, overlap)
                    else:
                        wiki_texts.append(text)
                        wiki_summaries.append(f"startseq {summary} endseq")
                        wiki_count += 1
                else:
                    logger.debug("Skipping article %d: empty text or summary", i)
                i += 1
            
            if wiki_count < wiki_samples:
                logger.warning("Collected only %d Wikipedia samples instead of %d", wiki_count, wiki_samples)
            
            wiki_df = pd.DataFrame({'text': wiki_texts, 'summary': wiki_summaries})
            
            # Combine datasets
            df = pd.concat([cnn_df, wiki_df], ignore_index=True)
            texts = df['text'].values
            summaries = df['summary'].values
            logger.info("Dataset loaded successfully: %d articles (1000 CNN/Daily Mail, %d Wikipedia)", len(texts), len(wiki_texts))
            return texts, summaries
        except Exception as e:
            logger.error("Failed to load dataset: %s", e)
            raise ValueError(f"Failed to load dataset: {e}")
    
    def tokenize_english(self, texts):
        """Tokenizes English text using NLTK."""
        logger.debug("Tokenizing %d texts", len(texts))
        return [nltk.word_tokenize(text.lower()) for text in texts]
    
    def prepare_data(self, texts, summaries, fit_tokenizer=True):
        """Converts texts and summaries to padded sequences."""
        logger.info("Preparing data for %d texts", len(texts))
        texts = texts.tolist() if isinstance(texts, np.ndarray) else texts
        if len(texts) == 0 or not all(isinstance(t, str) for t in texts):
            logger.error("Invalid input: texts must be non-empty strings")
            raise ValueError("Invalid input: texts must be non-empty strings")
        
        text_tokens = self.tokenize_english(texts)
        summary_tokens = self.tokenize_english(summaries) if summaries is not None and len(summaries) > 0 else [[]] * len(texts)
        
        if fit_tokenizer:
            self.text_tokenizer.fit_on_texts([' '.join(tokens) for tokens in text_tokens])
            self.summary_tokenizer.fit_on_texts([' '.join(tokens) for tokens in summary_tokens])
            logger.debug("Tokenizers fitted: text_vocab=%d, summary_vocab=%d", 
                        len(self.text_tokenizer.word_index) + 1, len(self.summary_tokenizer.word_index) + 1)
        
        x = self.text_tokenizer.texts_to_sequences([' '.join(tokens) for tokens in text_tokens])
        y = self.summary_tokenizer.texts_to_sequences([' '.join(tokens) for tokens in summary_tokens]) if summary_tokens and any(len(tokens) > 0 for tokens in summary_tokens) else [[]]
        
        # Validate token indices
        x_voc = len(self.text_tokenizer.word_index) + 1
        y_voc = len(self.summary_tokenizer.word_index) + 1
        x_max = max([max(seq) for seq in x if seq]) if x and any(seq for seq in x) else 0
        y_max = max([max(seq) for seq in y if seq]) if y and any(seq for seq in y) else 0
        logger.debug("Token indices: x_max=%d (x_voc=%d), y_max=%d (y_voc=%d)", x_max, x_voc, y_max, y_voc)
        if x_max >= x_voc:
            logger.error("Text sequences contain indices >= x_voc: %d >= %d", x_max, x_voc)
            raise ValueError(f"Text sequences contain indices >= x_voc: {x_max} >= {x_voc}")
        if y_max >= y_voc:
            logger.error("Summary sequences contain indices >= y_voc: %d >= %d", y_max, y_voc)
            raise ValueError(f"Summary sequences contain indices >= y_voc: {y_max} >= {y_voc}")
        
        x = pad_sequences(x, maxlen=self.max_text_len, padding='post', truncating='post')
        y = pad_sequences(y, maxlen=self.max_summary_len, padding='post', truncating='post') if summary_tokens and any(len(tokens) > 0 for tokens in summary_tokens) else None
        
        logger.info("Data prepared: x_shape=%s, y_shape=%s", str(x.shape), str(y.shape) if y is not None else "None")
        return x, y
    
    def split_dataset(self, texts, summaries):
        """Splits dataset into train, validation, and test sets."""
        logger.info("Splitting dataset: %d samples", len(texts))
        x_train, x_temp, y_train, y_temp = train_test_split(texts, summaries, test_size=0.2, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
        logger.info("Dataset split: train=%d, val=%d, test=%d", len(x_train), len(x_val), len(x_test))
        return x_train, x_val, x_test, y_train, y_val, y_test