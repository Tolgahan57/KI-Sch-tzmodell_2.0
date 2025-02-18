import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


class DataProcessor:
    def __init__(self, tokenizer, max_length=128, max_tfidf_features=100):
        """
        Initialisiert den DataProcessor.

        Args:
            tokenizer: Ein vorinitialisierter Tokenizer (z. B. von Hugging Face), der zur Tokenisierung verwendet wird.
            max_length: Maximale Länge der tokenisierten Sequenzen.
            max_tfidf_features: Maximale Anzahl an TF-IDF-Features für die Skills-Vektorisierung.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        # TF-IDF Vektorisierer speziell für die Skills
        self.tfidf_vectorizer_skills = TfidfVectorizer(max_features=max_tfidf_features)

    @staticmethod
    def load_data(filepath):
        """
        Lädt eine CSV-Datei und berücksichtigt unterschiedliche Encodings.

        Args:
            filepath (str): Pfad zur CSV-Datei.

        Returns:
            DataFrame: Geladene Daten.
        """
        try:
            data = pd.read_csv(filepath, delimiter=";", encoding="utf-8")
        except UnicodeDecodeError:
            data = pd.read_csv(filepath, delimiter=";", encoding="ISO-8859-1")
        return data

    @staticmethod
    def safe_str(value):
        """
        Konvertiert einen Wert in einen String, sofern er nicht fehlend ist.

        Args:
            value: Beliebiger Wert.

        Returns:
            str: Bereinigter String oder ein leerer String, wenn der Wert fehlt.
        """
        return str(value).strip() if pd.notna(value) else ""

    def extract_experience(self, row):
        """
        Extrahiert aus einer Zeile alle Einträge zur Berufserfahrung.
        Erwartet werden Spalten wie: position_Exp_i, company_name_Exp_i, duration_Exp_i (i = 1..10).

        Args:
            row (Series): Eine Zeile des DataFrames.

        Returns:
            str: Zusammengeführte Darstellung der Berufserfahrung.
        """
        experiences = []
        for i in range(1, 11):
            position = self.safe_str(row.get(f"position_Exp_{i}", ""))
            company = self.safe_str(row.get(f"company_name_Exp_{i}", ""))
            duration = self.safe_str(row.get(f"duration_Exp_{i}", ""))
            if position:
                experiences.append(f"{position} at {company} ({duration})".strip())
        return " ; ".join(experiences)

    def extract_education(self, row):
        """
        Extrahiert aus einer Zeile alle Einträge zur Ausbildung.
        Erwartet werden Spalten wie: college_degree_education_i, college_degree_field_education_i, college_name_education_i (i = 1..5).

        Args:
            row (Series): Eine Zeile des DataFrames.

        Returns:
            str: Zusammengeführte Darstellung der Ausbildung.
        """
        education = []
        for i in range(1, 6):
            degree = self.safe_str(row.get(f"college_degree_education_{i}", ""))
            field = self.safe_str(row.get(f"college_degree_field_education_{i}", ""))
            college = self.safe_str(row.get(f"college_name_education_{i}", ""))
            if college:
                education.append(f"{degree} in {field} from {college}".strip())
        return " ; ".join(education)

    def extract_skills(self, row):
        """
        Extrahiert die Skills aus einer Zeile.
        Erwartet wird die Spalte 'skills', in der mehrere Skills durch Kommata getrennt sind.

        Args:
            row (Series): Eine Zeile des DataFrames.

        Returns:
            str: Skills als bereinigter, durch " ; " getrennter String.
        """
        skills_str = self.safe_str(row.get("skills", ""))
        # Trennt anhand von Komma, entfernt leere Einträge und bereinigt die einzelnen Werte
        skills_list = [skill.strip() for skill in skills_str.split(",") if skill.strip()]
        return " ; ".join(skills_list)

    def extract_languages(self, row):
        """
        Extrahiert die Sprachen aus einer Zeile.
        Erwartet wird die Spalte 'Languages', in der mehrere Sprachen durch Kommata getrennt sind.

        Args:
            row (Series): Eine Zeile des DataFrames.

        Returns:
            str: Sprachen als bereinigter, durch " ; " getrennter String.
        """
        languages_str = self.safe_str(row.get("Languages", ""))
        languages_list = [lang.strip() for lang in languages_str.split(",") if lang.strip()]
        return " ; ".join(languages_list)

    def transform_mbti_label(self, label):
        """
        Transformiert einen MBTI-String (z. B. "INTJ") in einen 4-dimensionalen binären Vektor.

        Beispielhafte Kodierung:
          - E/I: E -> 1, I -> 0
          - S/N: N -> 1, S -> 0
          - T/F: T -> 1, F -> 0
          - J/P: J -> 1, P -> 0

        Args:
            label (str): MBTI-Typ als String.

        Returns:
            list: Liste von 4 binären Werten.
        """
        mapping = {
            'E': 1, 'I': 0,
            'N': 1, 'S': 0,
            'T': 1, 'F': 0,
            'J': 1, 'P': 0
        }
        label = label.upper().strip()
        if len(label) < 4:
            label = label.ljust(4, '0')
        return [mapping.get(ch, 0) for ch in label[:4]]

    def preprocess(self, df, fit_vectorizer=False):
        """
        Verarbeitet den gesamten DataFrame:
          - Füllt fehlende Werte.
          - Extrahiert separate Texte für Erfahrung, Ausbildung, Skills und Sprachen.
          - Tokenisiert Erfahrung, Ausbildung und Sprachen mit dem BERT-Tokenizer (padding="max_length").
          - Vektorisiert die Skills mit einem TF-IDF-Vektorisierer.
          - Transformiert die MBTI-Labels in einen 4-dimensionalen Vektor.

        Args:
            df (DataFrame): Eingabedaten.
            fit_vectorizer (bool): Wenn True, wird der TF-IDF-Vektorisierer an den Skills-Daten gefittet.
                                    Beim Training sollte dieser Parameter True sein, beim Anwenden auf Testdaten False.

        Returns:
            tuple: (X, y) wobei X ein Dictionary mit den folgenden Keys ist:
                   - "experience_input_ids", "experience_attention_mask" (BERT für Erfahrung)
                   - "education_input_ids", "education_attention_mask" (BERT für Ausbildung)
                   - "languages_input_ids", "languages_attention_mask" (BERT für Sprachen)
                   - "skills_tfidf" (TF-IDF Vektoren für Skills)
                   und y (falls vorhanden) die 4-dimensionalen MBTI-Labels als NumPy-Array enthält.
        """
        df = df.fillna("")

        # Extraktion der Texte
        df["experience_text"] = df.apply(self.extract_experience, axis=1)
        df["education_text"] = df.apply(self.extract_education, axis=1)
        df["skills_text"] = df.apply(self.extract_skills, axis=1)
        df["languages_text"] = df.apply(self.extract_languages, axis=1)

        # BERT-Tokenisierung für Erfahrung, Ausbildung und Sprachen
        tokenized_exp = self.tokenizer(
            df["experience_text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )
        tokenized_edu = self.tokenizer(
            df["education_text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )
        tokenized_lang = self.tokenizer(
            df["languages_text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )

        # TF-IDF Vektorisierung für Skills
        skills_texts = df["skills_text"].tolist()
        if fit_vectorizer:
            skills_tfidf = self.tfidf_vectorizer_skills.fit_transform(skills_texts).toarray()
        else:
            skills_tfidf = self.tfidf_vectorizer_skills.transform(skills_texts).toarray()

        # Zusammenstellen des Input-Dictionarys
        X = {
            "experience_input_ids": tokenized_exp["input_ids"],
            "experience_attention_mask": tokenized_exp["attention_mask"],
            "education_input_ids": tokenized_edu["input_ids"],
            "education_attention_mask": tokenized_edu["attention_mask"],
            "languages_input_ids": tokenized_lang["input_ids"],
            "languages_attention_mask": tokenized_lang["attention_mask"],
            "skills_tfidf": skills_tfidf
        }

        # Transformieren der MBTI-Labels (falls vorhanden)
        y = None
        if "MBTI" in df.columns:
            df["mbti_vector"] = df["MBTI"].apply(self.transform_mbti_label)
            y = np.array(df["mbti_vector"].tolist())

        return X, y
