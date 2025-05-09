from rouge_score import rouge_scorer
from utils.logger import setup_logger

logger = setup_logger('logs/app.log')

class Evaluator:
    """Handles evaluation of generated summaries using ROUGE scores."""
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        logger.info("Evaluator initialized")
    
    def calculate_rouge(self, generated_summary, reference_summary):
        """Calculates ROUGE scores between generated and reference summaries."""
        logger.debug("Calculating ROUGE scores for generated summary: %s", generated_summary[:50])
        scores = self.scorer.score(reference_summary, generated_summary)
        result = {
            'rouge-1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'f1': scores['rouge1'].fmeasure
            },
            'rouge-2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'f1': scores['rouge2'].fmeasure
            },
            'rouge-l': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'f1': scores['rougeL'].fmeasure
            }
        }
        logger.debug("ROUGE scores: %s", result)
        return result