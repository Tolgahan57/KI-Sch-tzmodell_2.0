import os
import tensorflow as tf
from transformers import AutoTokenizer
from modules.DataProcessor import DataProcessor
from modules.MBTIModule import MBTIModule
from modules.MBTIPredictor import MBTIPredictor

# GPU-Konfiguration: GPU auswählen und Speicherwachstum aktivieren
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Wähle die erste verfügbare GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU erkannt und konfiguriert!")
    except RuntimeError as e:
        print(f"⚠️ Fehler bei der GPU-Konfiguration: {e}")
else:
    print("⚠️ Keine GPU erkannt. Das Training läuft auf der CPU.")

# Pfad zur CSV-Datei mit den Trainingsdaten
file_path = r"C:\Users\Tolgahan\Downloads\LinkedIn-personalities5.CSV"


def main():
    # 1. Tokenizer initialisieren
    print("🔧 Initialisiere den Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # 2. DataProcessor initialisieren und Daten laden & vorverarbeiten
    print("📥 Lade und preprocess die Daten...")
    processor = DataProcessor(tokenizer=tokenizer, max_length=128)
    data = processor.load_data(file_path)
    X, y = processor.preprocess(data)
    print("✅ Daten erfolgreich vorverarbeitet!")
    print("   Formen der Inputs:")
    for key, value in X.items():
        print(f"   {key}: {value.shape}")
    if y is not None:
        print(f"   Labels (y): {y.shape}")

    # 3. Modell laden oder neu erstellen und trainieren
    model_path = "mbti_module.keras"
    if os.path.exists(model_path):
        print("📂 Vorhandenes Modell gefunden. Lade Modell...")
        model = MBTIModule.load(model_path)
    else:
        print("🚀 Kein Modell gefunden. Erstelle und trainiere ein neues Modell...")
        mbti_module = MBTIModule(
            model_name="bert-base-multilingual-cased",
            maxlen=128,
            epochs=5,
            batch_size=16,
            learning_rate=2e-5
        )
        mbti_module.train(X, y)
        mbti_module.save(model_path)
        model = mbti_module.model

    # 4. MBTI_Predictor initialisieren
    print("🔮 Initialisiere den MBTI-Predictor...")
    predictor = MBTIPredictor(model=model, tokenizer_name="bert-base-multilingual-cased", max_length=128)

    # 5. Test: Vorhersage für ein neues Profil
    test_profile = (
        """
    Team Lead Purchasing at AMW GmbH (11 months)
    Director Business Partnerships at Image Professionals (1 year 2 months)
    Head of Procurement at Hubert Burda Media (3 years 5 months)
    Master of Business Administration - Procurement and Sourcing (2015 - 2017)
    German, English, Spanish
    Procurement, Stakeholder Management, Business Development, Key Account Management
    """"
    )
    predicted_mbti = predictor.predict(test_profile)
    print(f"🎯 Vorhergesagter MBTI-Typ: {predicted_mbti}")


if __name__ == "__main__":
    main()
