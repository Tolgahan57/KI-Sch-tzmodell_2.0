import tensorflow as tf
from transformers import TFAutoModel
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.optimizers import Adam


class MBTIModule:
    def __init__(self,
                 model_name="bert-base-multilingual-cased",
                 maxlen=128,
                 max_tfidf_features=100,
                 epochs=5,
                 batch_size=16,
                 learning_rate=2e-5,
                 dense_units=256,
                 dropout_rate=0.2):
        """
        Initialisiert das MBTI-Modul mit separaten Inputs für:
          - Erfahrung, Ausbildung und Sprachen (tokenisiert mit BERT)
          - Skills (als TF-IDF-Vektor)
        Die Ausgabe besteht aus 4 Sigmoid-Aktivierungen (eine pro MBTI-Dichotomie).

        Args:
            model_name (str): Name des vortrainierten BERT-Modells.
            maxlen (int): Maximale Länge der tokenisierten Sequenzen.
            max_tfidf_features (int): Dimension des TF-IDF-Vektors für Skills.
            epochs (int): Anzahl der Trainingsepochen.
            batch_size (int): Batch-Größe.
            learning_rate (float): Lernrate des Optimierers.
            dense_units (int): Anzahl der Neuronen in der ersten Dense-Schicht (tunable).
            dropout_rate (float): Dropout-Rate (tunable).
        """
        self.model_name = model_name
        self.maxlen = maxlen
        self.max_tfidf_features = max_tfidf_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        self.build_model()

    def build_model(self):
        # Definiert Inputs für die BERT-basierten Felder:
        input_ids_exp = Input(shape=(self.maxlen,), dtype=tf.int32, name="experience_input_ids")
        attention_mask_exp = Input(shape=(self.maxlen,), dtype=tf.int32, name="experience_attention_mask")

        input_ids_edu = Input(shape=(self.maxlen,), dtype=tf.int32, name="education_input_ids")
        attention_mask_edu = Input(shape=(self.maxlen,), dtype=tf.int32, name="education_attention_mask")

        input_ids_lang = Input(shape=(self.maxlen,), dtype=tf.int32, name="languages_input_ids")
        attention_mask_lang = Input(shape=(self.maxlen,), dtype=tf.int32, name="languages_attention_mask")

        # Skills Input: TF-IDF Vektor
        skills_tfidf_input = Input(shape=(self.max_tfidf_features,), dtype=tf.float32, name="skills_tfidf")

        # BERT-Modell laden (gemeinsam für alle textlichen Inputs)
        bert_model = TFAutoModel.from_pretrained(self.model_name)
        embedding_exp = bert_model(input_ids_exp, attention_mask=attention_mask_exp)[1]
        embedding_edu = bert_model(input_ids_edu, attention_mask=attention_mask_edu)[1]
        embedding_lang = bert_model(input_ids_lang, attention_mask=attention_mask_lang)[1]

        # Transformation des Skills-Inputs über eine Dense-Schicht
        skills_dense = Dense(128, activation="relu", name="skills_dense")(skills_tfidf_input)

        # Alle Kanäle zusammenführen
        combined = Concatenate(name="combined_embeddings")([embedding_exp, embedding_edu, embedding_lang, skills_dense])

        # Klassifikationskopf mit den tunebaren Parametern
        x = Dense(self.dense_units, activation="relu", name="dense_1")(combined)
        x = Dropout(self.dropout_rate, name="dropout")(x)
        output = Dense(4, activation="sigmoid", name="output")(x)

        self.model = Model(
            inputs=[input_ids_exp, attention_mask_exp,
                    input_ids_edu, attention_mask_edu,
                    input_ids_lang, attention_mask_lang,
                    skills_tfidf_input],
            outputs=output
        )
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        print("Modell erfolgreich erstellt!")

    def train(self, X, y):
        """
        Trainiert das MBTI-Modell.

        Args:
            X (dict): Dictionary mit den folgenden Keys:
                - "experience_input_ids", "experience_attention_mask"
                - "education_input_ids", "education_attention_mask"
                - "languages_input_ids", "languages_attention_mask"
                - "skills_tfidf"
            y (numpy.array): Array der Form (n_samples, 4) für die MBTI-Dichotomien.
        """
        print("Starte Training...")
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)
        print("Training abgeschlossen!")

    def save(self, model_path="mbti_module.keras"):
        """
        Speichert das trainierte Modell.
        """
        self.model.save(model_path)
        print(f"Modell gespeichert unter: {model_path}")

    @staticmethod
    def load(model_path):
        """
        Lädt ein gespeichertes Modell.
        """
        print(f"Lade Modell aus {model_path}...")
        return tf.keras.models.load_model(model_path)
