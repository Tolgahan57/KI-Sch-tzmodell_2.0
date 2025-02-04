from transformers import AutoTokenizer

class MBTIPredictor:
    def __init__(self, model, tokenizer_name="bert-base-multilingual-cased", max_length=128):
        """
        Initialisiert den MBTI-Predictor.

        Args:
            model: Das trainierte MBTI-Modell.
            tokenizer_name (str): Name des Tokenizers, der zur Tokenisierung verwendet wird.
            max_length (int): Maximale Länge der tokenisierten Sequenzen.
        """
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
