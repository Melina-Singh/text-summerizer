import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Dropout
from tensorflow.keras.layers import Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from utils.logger import setup_logger

logger = setup_logger('logs/app.log')

class Seq2SeqModel:
    """Seq2Seq model with GloVe embeddings, bidirectional LSTM encoder, and LSTM decoder with attention."""
    
    def __init__(self, x_voc, y_voc, max_text_len, max_summary_len, glove_embedding_matrix, glove_dim=50, latent_dim=200):
        self.x_voc = x_voc
        self.y_voc = y_voc
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.glove_embedding_matrix = glove_embedding_matrix
        self.glove_dim = glove_dim
        self.latent_dim = latent_dim
        self.banned_tokens = ['genocide', 'heineken', 'terrorism', 'war', 'election', 'summit', 
                             'minister', 'attack', 'crisis', 'protest', 'newterm']
        logger.info("Seq2SeqModel initialized: x_voc=%d, y_voc=%d, latent_dim=%d, glove_dim=%d", 
                    x_voc, y_voc, latent_dim, glove_dim)
    
    def build_model(self):
        """Builds the seq2seq model with attention."""
        # Encoder
        encoder_inputs = Input(shape=(self.max_text_len,))
        encoder_embedding = Embedding(
            self.x_voc, 
            self.glove_dim, 
            weights=[self.glove_embedding_matrix], 
            trainable=False
        )(encoder_inputs)
        encoder_dropout = Dropout(0.3)(encoder_embedding)
        encoder = Bidirectional(LSTM(self.latent_dim, return_sequences=True, return_state=True))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_dropout)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        
        # Decoder
        decoder_inputs = Input(shape=(self.max_summary_len-1,))
        decoder_embedding_layer = Embedding(self.y_voc, self.glove_dim)
        decoder_embedding = decoder_embedding_layer(decoder_inputs)
        decoder_dropout = Dropout(0.3)(decoder_embedding)
        decoder_lstm = LSTM(self.latent_dim*2, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_dropout, initial_state=[state_h, state_c])
        
        # Attention
        attention = Attention()
        context_vector = attention([decoder_outputs, encoder_outputs])
        decoder_combined_context = Concatenate()([decoder_outputs, context_vector])
        
        # Dense output
        decoder_dense = Dense(self.y_voc, activation='softmax')
        decoder_outputs = decoder_dense(decoder_combined_context)
        
        # Define training model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Compile with gradient clipping
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
        logger.info("Training model compiled")
        
        # Inference models
        self.encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])
        
        decoder_state_input_h = Input(shape=(self.latent_dim*2,))
        decoder_state_input_c = Input(shape=(self.latent_dim*2,))
        decoder_outputs_input = Input(shape=(self.max_text_len, self.latent_dim*2))
        dec_emb = decoder_embedding_layer(decoder_inputs)
        dec_emb_dropout = Dropout(0.3)(dec_emb)
        decoder_outputs, state_h, state_c = decoder_lstm(dec_emb_dropout, 
                                                        initial_state=[decoder_state_input_h, decoder_state_input_c])
        context_vector = attention([decoder_outputs, decoder_outputs_input])
        decoder_combined_context = Concatenate()([decoder_outputs, context_vector])
        decoder_outputs = decoder_dense(decoder_combined_context)
        
        self.decoder_model = Model(
            [decoder_inputs, decoder_outputs_input, decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs, state_h, state_c]
        )
        logger.info("Inference models built")
    
    def generate_summary(self, input_seq, summary_tokenizer, max_summary_len, beam_width=5):
        """Generates summary using beam search."""
        logger.info("Generating summary for input sequence")
        e_out, h, c = self.encoder_model.predict(input_seq, verbose=0)
        
        # Initialize beam search
        sequences = [[[], 0.0, h, c]]
        for _ in range(max_summary_len):
            all_candidates = []
            for seq, score, h, c in sequences:
                if seq and seq[-1] == summary_tokenizer.word_index.get('endseq', 0):
                    all_candidates.append([seq, score, h, c])
                    continue
                target_seq = np.zeros((1, max_summary_len-1))
                target_seq[0, 0] = seq[-1] if seq else summary_tokenizer.word_index.get('startseq', 1)
                
                output_tokens, h_new, c_new = self.decoder_model.predict(
                    [target_seq, e_out, h, c], verbose=0
                )
                probs = output_tokens[0, -1, :]
                
                # Ban specific tokens
                for token in self.banned_tokens:
                    if token in summary_tokenizer.word_index:
                        probs[summary_tokenizer.word_index[token]] = 0
                probs /= probs.sum() + 1e-10
                
                # Get top beam_width tokens
                top_indices = np.argsort(probs)[-beam_width:]
                for idx in top_indices:
                    new_seq = seq + [idx]
                    new_score = score + np.log(probs[idx] + 1e-10)
                    all_candidates.append([new_seq, new_score, h_new, c_new])
            
            # Select top beam_width sequences
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        best_seq = sequences[0][0]
        summary = ' '.join([summary_tokenizer.index_word.get(idx, '') for idx in best_seq 
                           if idx in summary_tokenizer.index_word and 
                           summary_tokenizer.index_word[idx] not in ['startseq', 'endseq']])
        logger.debug("Generated summary: %s", summary)
        return summary