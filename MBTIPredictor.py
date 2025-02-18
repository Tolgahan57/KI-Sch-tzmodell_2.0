import numpy as np
from transformers import AutoTokenizer


class MBTIPredictor:
    def __init__(self, model, tokenizer_name="bert-base-multilingual-cased", max_length=128, tfidf_vectorizer=None):
        """
        Initialisiert den MBTI-Predictor.

        Args:
            model: Das trainierte MBTI-Modell.
            tokenizer_name (str): Name des Tokenizers (z. B. "bert-base-multilingual-cased").
            max_length (int): Maximale Länge der tokenisierten Sequenzen.
            tfidf_vectorizer: Ein bereits gefitteter TF-IDF-Vektorisierer, der für die Skills verwendet wird.
                              Dieser Vektorisierer muss während des Trainings (im DataProcessor) gefittet worden sein.
        """
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        if tfidf_vectorizer is None:
            raise ValueError("Ein gefitteter TF-IDF-Vektorisierer muss übergeben werden!")
        self.tfidf_vectorizer = tfidf_vectorizer

    def preprocess_text(self, text_data):
        """
        Bereitet die Eingabedaten für die Inferenz vor.

        Erwartet wird idealerweise ein Dictionary mit den Schlüsseln:
            "experience_text", "education_text", "languages_text" und "skills_text".
        Fehlen einige Keys, so werden diese automatisch mit einem leeren String belegt.

        Falls text_data ein einzelner String ist, wird dieser als Fallback in alle Felder kopiert.

        Args:
            text_data (str oder dict): Eingabetext oder Dictionary mit den Textfeldern.

        Returns:
            dict: Dictionary mit den folgenden Keys:
                - "experience_input_ids", "experience_attention_mask"
                - "education_input_ids", "education_attention_mask"
                - "languages_input_ids", "languages_attention_mask"
                - "skills_tfidf" (als TF-IDF-Vektor)
        """
        # Falls text_data ein einzelner String ist, wird er in alle Felder übernommen.
        if not isinstance(text_data, dict):
            print("Warnung: Es wurde kein Dictionary übergeben. Verwende denselben Text für alle Felder!")
            text_data = {
                "experience_text": text_data,
                "education_text": text_data,
                "languages_text": text_data,
                "skills_text": text_data
            }
        else:
            # Stellt sicher, dass alle erforderlichen Keys vorhanden sind.
            required_keys = ["experience_text", "education_text", "languages_text", "skills_text"]
            for key in required_keys:
                if key not in text_data:
                    # Fehlende Keys werden mit einem leeren String belegt.
                    text_data[key] = ""

        # Tokenisierung der textlichen Felder
        tokenized_exp = self.tokenizer(
            [text_data["experience_text"]],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )
        tokenized_edu = self.tokenizer(
            [text_data["education_text"]],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )
        tokenized_lang = self.tokenizer(
            [text_data["languages_text"]],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )

        # Skills: TF-IDF Vektorisierung (mit dem bereits gefitteten Vektorisierer)
        skills_tfidf = self.tfidf_vectorizer.transform([text_data["skills_text"]]).toarray()

        inputs = {
            "experience_input_ids": tokenized_exp["input_ids"],
            "experience_attention_mask": tokenized_exp["attention_mask"],
            "education_input_ids": tokenized_edu["input_ids"],
            "education_attention_mask": tokenized_edu["attention_mask"],
            "languages_input_ids": tokenized_lang["input_ids"],
            "languages_attention_mask": tokenized_lang["attention_mask"],
            "skills_tfidf": skills_tfidf
        }
        return inputs

    def predict(self, text_data):
        """
        Gibt eine MBTI-Vorhersage für das übergebene Profil zurück.

        Args:
            text_data (str oder dict): Entweder ein einzelner String oder ein Dictionary mit den Schlüsseln
                                       "experience_text", "education_text", "languages_text" und "skills_text".

        Returns:
            str: Der vorhergesagte MBTI-Typ (z. B. "INTJ").
        """
        inputs = self.preprocess_text(text_data)
        predictions = self.model.predict(inputs)  # Erwartet: shape (batch_size, 4)
        pred_vector = predictions[0]

        # Anwenden eines Schwellenwerts von 0.5: Für jeden Wert >= 0.5 wird der positive Pol ausgewählt, sonst der negative.
        letters_positive = ['E', 'N', 'T', 'J']
        letters_negative = ['I', 'S', 'F', 'P']
        mbti_type = ""
        for i, prob in enumerate(pred_vector):
            mbti_type += letters_positive[i] if prob >= 0.5 else letters_negative[i]
        return mbti_type
