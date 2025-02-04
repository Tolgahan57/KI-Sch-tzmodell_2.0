import tensorflow as tf
from transformers import TFAutoModel
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.optimizers import Adam


class MBTIModule:
    def __init__(self,
                 model_name="bert-base-multilingual-cased",
                 maxlen=128,
                 epochs=5,
                 batch_size=16,
                 learning_rate=2e-5):
        """
        Initialisiert das MBTI-Modul, das vier binäre Vorhersagen (E/I, S/N, T/F, J/P) ausgibt.

        Args:
            model_name (str): Name des vortrainierten BERT-Modells.
            maxlen (int): Maximale Länge der Eingabesequenzen.
            epochs (int): Anzahl der Trainingsepochen.
            batch_size (int): Batch-Größe.
            learning_rate (float): Lernrate des Optimierers.
        """
        self.model_name = model_name
        self.maxlen = maxlen
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.build_model()

    def build_model(self):
        """
        Erstellt das MBTI-Modell mit separaten Eingabekanälen für Erfahrung, Ausbildung,
        Skills und Sprachen. Die Ausgaben erfolgen als vier binäre Vorhersagen für die
        Dichotomien (E/I, S/N, T/F, J/P).
        """
        print("Erstelle das MBTI-Modell mit separaten Eingabekanälen und dichotomen Outputs...")

        # Separate Eingabeschichten für jeden Datenkanal
        input_ids_exp = Input(shape=(self.maxlen,), dtype=tf.int32, name="experience_input_ids")
        attention_mask_exp = Input(shape=(self.maxlen,), dtype=tf.int32, name="experience_attention_mask")

        input_ids_edu = Input(shape=(self.maxlen,), dtype=tf.int32, name="education_input_ids")
        attention_mask_edu = Input(shape=(self.maxlen,), dtype=tf.int32, name="education_attention_mask")

        input_ids_skills = Input(shape=(self.maxlen,), dtype=tf.int32, name="skills_input_ids")
        attention_mask_skills = Input(shape=(self.maxlen,), dtype=tf.int32, name="skills_attention_mask")

        input_ids_lang = Input(shape=(self.maxlen,), dtype=tf.int32, name="languages_input_ids")
        attention_mask_lang = Input(shape=(self.maxlen,), dtype=tf.int32, name="languages_attention_mask")

        # Vortrainiertes BERT-Modell laden (gemeinsam für alle Kanäle)
        bert_model = TFAutoModel.from_pretrained(self.model_name)

        # Für jeden Kanal den pooled output extrahieren
        embedding_exp = bert_model(input_ids_exp, attention_mask=attention_mask_exp)[1]
        embedding_edu = bert_model(input_ids_edu, attention_mask=attention_mask_edu)[1]
        embedding_skills = bert_model(input_ids_skills, attention_mask=attention_mask_skills)[1]
        embedding_lang = bert_model(input_ids_lang, attention_mask=attention_mask_lang)[1]

        # Alle Embeddings zusammenführen
        combined = Concatenate(name="combined_embeddings")([
            embedding_exp,
            embedding_edu,
            embedding_skills,
            embedding_lang
        ])

        # Gemeinsamer Klassifikationskopf
        x = Dense(256, activation="relu", name="dense_1")(combined)
        x = Dropout(0.2, name="dropout")(x)

        # Output-Layer: 4 Neuronen, jeweils eine binäre Vorhersage per Sigmoid
        output = Dense(4, activation="sigmoid", name="output")(x)

        # Modell definieren
        self.model = Model(
            inputs=[
                input_ids_exp, attention_mask_exp,
                input_ids_edu, attention_mask_edu,
                input_ids_skills, attention_mask_skills,
                input_ids_lang, attention_mask_lang
            ],
            outputs=output
        )

        # Modell kompilieren: binary_crossentropy für 4 unabhängige Binärklassifikationen
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("Modell erfolgreich erstellt!")

    def train(self, X, y):
        """
        Trainiert das MBTI-Modell.

            X (dict): Dictionary mit tokenisierten Daten, mit den Keys:
                      - "experience_input_ids", "experience_attention_mask",
                      - "education_input_ids", "education_attention_mask",
                      - "skills_input_ids", "skills_attention_mask",
                      - "languages_input_ids", "languages_attention_mask"
            y (numpy.array): Array mit Shape (n_samples, 4), das die vier binären Zielwerte
                             für E/I, S/N, T/F und J/P enthält.
        """
        print("Starte Training...")
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1
        )
        print("Training abgeschlossen!")

    def save(self, model_path="mbti_module.keras"):
        """
        Speichert das trainierte Modell.

        Args:
            model_path (str): Pfad, unter dem das Modell gespeichert wird.
        """
        self.model.save(model_path)
        print(f"Modell gespeichert unter: {model_path}")

    @staticmethod
    def load(model_path):
        """
        Lädt ein gespeichertes Modell.

        Args:
            model_path (str): Pfad zum gespeicherten Modell.

        Returns:
            tf.keras.Model: Das geladene Modell.
        """
        print(f"Lade Modell aus {model_path}...")
        return tf.keras.models.load_model(model_path)
