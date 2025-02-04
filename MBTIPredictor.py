import numpy as np
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

    def preprocess_text(self, text_data):
        """
        Tokenisiert den übergebenen Text und erstellt ein Dictionary, das
        alle erforderlichen Eingaben für das MBTI-Modell enthält.

        Da das Modell vier Eingabekanäle erwartet (experience, education, skills, languages),
        verwenden wir hier als ersten Prototyp denselben tokenisierten Text für alle Kanäle.

        Args:
            text_data (str or list): Der Eingabetext oder eine Liste von Texten.

        Returns:
            dict: Dictionary mit den Keys:
                - "experience_input_ids", "experience_attention_mask",
                - "education_input_ids", "education_attention_mask",
                - "skills_input_ids", "skills_attention_mask",
                - "languages_input_ids", "languages_attention_mask"
        """
        # Falls text_data ein einzelner String ist, packe ihn in eine Liste
        if isinstance(text_data, str):
            text_data = [text_data]
        tokenized = self.tokenizer(
            text_data,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )
        inputs = {
            "experience_input_ids": tokenized["input_ids"],
            "experience_attention_mask": tokenized["attention_mask"],
            "education_input_ids": tokenized["input_ids"],
            "education_attention_mask": tokenized["attention_mask"],
            "skills_input_ids": tokenized["input_ids"],
            "skills_attention_mask": tokenized["attention_mask"],
            "languages_input_ids": tokenized["input_ids"],
            "languages_attention_mask": tokenized["attention_mask"],
        }
        return inputs

    def predict(self, text_data):
        """
        Gibt eine MBTI-Vorhersage für das übergebene Profil zurück.

        Die Vorhersage erfolgt als vier binäre Werte, die dann in einen MBTI-Typ (z. B. "ENTJ")
        umgewandelt werden. Dabei wird für jeden Wert ein Schwellenwert von 0.5 verwendet.

        Args:
            text_data (str): Der Eingabetext des Profils.

        Returns:
            str: Der vorhergesagte MBTI-Typ.
        """
        inputs = self.preprocess_text(text_data)
        predictions = self.model.predict(inputs)  # Erwartete Form: (batch_size, 4)

        # Nehme den ersten Eintrag, falls mehrere übergeben wurden
        pred_vector = predictions[0]

        # Definiere die Mapping-Regeln für die vier Dichotomien
        # Für jeden Index: >= 0.5 → positiver Pol, sonst negativer Pol
        letters_positive = ['E', 'N', 'T', 'J']  # z.B. E, N, T, J
        letters_negative = ['I', 'S', 'F', 'P']  # z.B. I, S, F, P

        mbti_type = ""
        for i, prob in enumerate(pred_vector):
            if prob >= 0.5:
                mbti_type += letters_positive[i]
            else:
                mbti_type += letters_negative[i]
        return mbti_type
