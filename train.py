import numpy as np
from models.seq2seq_model import Seq2SeqModel
from utils.data_preprocessing import DataPreprocessor
from utils.evaluation import Evaluator
from utils.logger import setup_logger
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

logger = setup_logger('logs/app.log')

def train_model():
    logger.info("Starting training process")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(max_text_len=300, max_summary_len=50, glove_dim=50)
    
    # Load dataset
    texts, summaries = preprocessor.load_dataset(num_samples=2000)
    
    # Validate dataset size
    if len(texts) != 2000 or len(summaries) != 2000:
        logger.error("Expected 2000 samples, got %d texts and %d summaries", len(texts), len(summaries))
        raise ValueError(f"Expected 2000 samples, got {len(texts)} texts and {len(summaries)} summaries")
    
    # Split dataset
    x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.split_dataset(texts, summaries)
    
    # Prepare data
    x_train_seq, y_train_seq = preprocessor.prepare_data(x_train, y_train)
    x_val_seq, y_val_seq = preprocessor.prepare_data(x_val, y_val, fit_tokenizer=False)
    x_test_seq, y_test_seq = preprocessor.prepare_data(x_test, y_test, fit_tokenizer=False)
    
    # Load GloVe embeddings and create embedding matrix
    glove_file = 'glove.6B.50d.txt'  # Update path if needed
    preprocessor.load_glove_embeddings(glove_file)
    embedding_matrix = preprocessor.create_embedding_matrix()
    
    # Log vocabulary coverage for input text
    test_input = "Domestic animals are animals that have been tamed and kept by humans for companionship, work, or food. Common examples include dogs, cats, cows, goats, sheep, and chickens. These animals have adapted to living alongside humans over thousands of years through selective breeding."
    test_tokens = preprocessor.tokenize_english([test_input])[0]
    test_seq = preprocessor.text_tokenizer.texts_to_sequences([' '.join(test_tokens)])[0]
    unknown_tokens = [t for t in test_tokens if t not in preprocessor.text_tokenizer.word_index]
    logger.info("Input text vocabulary coverage: %d/%d tokens known, unknown: %s", 
                len(test_seq), len(test_tokens), unknown_tokens)
    
    # Initialize model
    model = Seq2SeqModel(
        x_voc=len(preprocessor.text_tokenizer.word_index) + 1,
        y_voc=len(preprocessor.summary_tokenizer.word_index) + 1,
        max_text_len=300,
        max_summary_len=50,
        glove_embedding_matrix=embedding_matrix,
        glove_dim=50,
        latent_dim=200
    )
    model.build_model()
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    checkpoint = ModelCheckpoint('models/saved/best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    
    # Train model with teacher forcing
    logger.info("Training model with teacher forcing")
    history = model.model.fit(
        [x_train_seq, y_train_seq[:, :-1]],
        y_train_seq[:, 1:].reshape(y_train_seq.shape[0], y_train_seq.shape[1]-1, 1),
        epochs=40,
        batch_size=8,
        validation_data=([x_val_seq, y_val_seq[:, :-1]], y_val_seq[:, 1:].reshape(y_val_seq.shape[0], y_val_seq.shape[1]-1, 1)),
        callbacks=[early_stopping, lr_scheduler, checkpoint],
        verbose=1
    )
    logger.info("Model training completed")
    
    # Evaluate model
    evaluator = Evaluator()
    logger.info("Evaluator initialized")
    
    rouge_scores = []
    for i in range(len(x_test_seq)):
        input_seq = x_test_seq[i:i+1]
        ref_summary = ' '.join([preprocessor.summary_tokenizer.index_word.get(idx, '') for idx in y_test_seq[i] if idx > 0])
        gen_summary = model.generate_summary(input_seq, preprocessor.summary_tokenizer, preprocessor.max_summary_len)
        scores = evaluator.calculate_rouge(gen_summary, ref_summary)
        rouge_scores.append(scores)
    
    # Average ROUGE scores
    avg_rouge = {
        'rouge-1': np.mean([s['rouge-1']['f1'] for s in rouge_scores]),
        'rouge-2': np.mean([s['rouge-2']['f1'] for s in rouge_scores]),
        'rouge-l': np.mean([s['rouge-l']['f1'] for s in rouge_scores])
    }
    logger.info("Evaluation completed: ROUGE Scores: %s", avg_rouge)
    
    # Test with user input
    test_seq = pad_sequences([test_seq], maxlen=300, padding='post', truncating='post')
    gen_summary = model.generate_summary(test_seq, preprocessor.summary_tokenizer, preprocessor.max_summary_len)
    logger.info("Test input summary: %s", gen_summary)
    
    # Save models and tokenizers
    os.makedirs('models/saved', exist_ok=True)
    model.model.save('models/saved/seq2seq_model.h5')
    model.encoder_model.save('models/saved/encoder_model.h5')
    model.decoder_model.save('models/saved/decoder_model.h5')
    
    import pickle
    with open('models/saved/text_tokenizer.pkl', 'wb') as f:
        pickle.dump(preprocessor.text_tokenizer, f)
    with open('models/saved/summary_tokenizer.pkl', 'wb') as f:
        pickle.dump(preprocessor.summary_tokenizer, f)
    
    logger.info("Models and tokenizers saved")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise