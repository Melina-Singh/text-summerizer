import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import os
from utils.logger import setup_logger
from utils.evaluation import evaluate_model

logger = setup_logger('logs/transformer.log')

class SummarizationDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_input_len=512, max_target_len=120):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        input_text = f"summarize: {text}"
        
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            summary,
            max_length=self.max_target_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

def train_transformer():
    logger.info("Starting transformer training process")
    
    try:
        cnn_dataset = load_dataset('cnn_dailymail', '3.0.0', split='train')
        wiki_dataset = load_dataset('wikipedia', '20220301.en', split='train')
        logger.info("Datasets loaded successfully")
    except Exception as e:
        logger.error("Failed to load datasets: %s", e)
        return
    
    num_samples = 3000  # Increased for better training
    cnn_df = pd.DataFrame({
        'text': cnn_dataset['article'][:1500],
        'summary': cnn_dataset['highlights'][:1500]
    })
    
    wiki_count = 0
    wiki_texts, wiki_summaries = [], []
    for i, article in enumerate(wiki_dataset):
        text = article['text'].split('\n')[0]
        if len(text.split()) > 50:
            wiki_texts.append(text)
            wiki_summaries.append(' '.join(text.split()[:30]))
            wiki_count += 1
        if wiki_count >= 1500:
            break
    
    wiki_df = pd.DataFrame({'text': wiki_texts, 'summary': wiki_summaries})
    dataset_df = pd.concat([cnn_df, wiki_df], ignore_index=True)
    dataset_df = dataset_df.sample(frac=1, random_state=42).reset_index(drop=True)[:num_samples]
    logger.info("Dataset prepared: %d samples", len(dataset_df))
    
    train_size = int(0.8 * len(dataset_df))
    train_df = dataset_df[:train_size]
    val_df = dataset_df[train_size:]
    
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    logger.info("T5 model and tokenizer initialized")
    
    train_dataset = SummarizationDataset(
        train_df['text'].tolist(),
        train_df['summary'].tolist(),
        tokenizer,
        max_input_len=512,
        max_target_len=120
    )
    val_dataset = SummarizationDataset(
        val_df['text'].tolist(),
        val_df['summary'].tolist(),
        tokenizer,
        max_input_len=512,
        max_target_len=120
    )
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,  # Increased for better training
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    try:
        logger.info("Starting model training")
        trainer.train()
        logger.info("Model training completed")
    except Exception as e:
        logger.error("Training failed: %s", e)
        return
    
    model_dir = 'models/saved'
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(os.path.join(model_dir, 't5_model'))
    tokenizer.save_pretrained(os.path.join(model_dir, 't5_tokenizer'))
    logger.info("Model and tokenizer saved to %s", model_dir)
    
    test_text = dataset_df['text'].iloc[0]
    test_summary = dataset_df['summary'].iloc[0]
    input_text = f"summarize: {test_text}"
    inputs = tokenizer(
        input_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    outputs = model.generate(
        input_ids=inputs['input_ids'].to(model.device),
        attention_mask=inputs['attention_mask'].to(model.device),
        max_length=120,
        min_length=20,
        length_penalty=0.4,
        num_beams=8,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Test input summary: %s", generated_summary)
    
    rouge_scores = evaluate_model([test_summary], [generated_summary])
    logger.info("Transformer evaluation completed: ROUGE Scores: %s", rouge_scores)

if __name__ == '__main__':
    train_transformer()