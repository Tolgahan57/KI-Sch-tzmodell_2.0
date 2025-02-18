import optuna
from transformers import AutoTokenizer
from modules.DataProcessor import DataProcessor
from modules.MBTIModule import MBTIModule
import pickle

def objective(trial):
    # Hyperparameter, die getuned werden, werden definiert
    max_length = trial.suggest_categorical("max_length", [128, 256])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-5)
    dense_units = trial.suggest_int("dense_units", 128, 512, step=64)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    max_tfidf_features = trial.suggest_categorical("max_tfidf_features", [50, 100, 150])

    # Initialisiere Tokenizer und DataProcessor
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    processor = DataProcessor(tokenizer=tokenizer, max_length=max_length, max_tfidf_features=max_tfidf_features)

    # Lade Trainingsdaten
    file_path = r"C:\Users\Tolgahan\Downloads\LinkedIn-personalities5.CSV"
    df = processor.load_data(file_path)
    # Fit den TF-IDF-Vektorisierer an den Trainingsdaten --> Dieser muss auf false, wenn wir mit dem Training durch sind
    X, y = processor.preprocess(df, fit_vectorizer=True)

    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(processor.tfidf_vectorizer_skills, f)

    # Erstelle das MBTIModule mit den getunten Hyperparametern
    mbti_module = MBTIModule(
        model_name="bert-base-multilingual-cased",
        maxlen=max_length,
        max_tfidf_features=max_tfidf_features,
        epochs=3,  # Für das Tuning kürzer trainieren
        batch_size=batch_size,
        learning_rate=learning_rate,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )

    # Trainiere das Modell (Verwende einen Validierungssplit von z.B. 10%)
    history = mbti_module.model.fit(
        X, y,
        epochs=3,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0
    )

    # Zielgröße: Validierungsverlust der letzten Epoche
    val_loss = history.history["val_loss"][-1]
    return val_loss


# Erstelle und starte die Optuna-Studie
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Beste Hyperparameter:")
print(study.best_params)
print("Besten Validierungsverlust:")
print(study.best_value)


# Nach dem Tuning kann man das Modell mit den besten Parametern final trainieren:
def main():
    # Nutzen die besten Hyperparameter
    best_params = study.best_params
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    processor = DataProcessor(tokenizer=tokenizer, max_length=best_params["max_length"],
                              max_tfidf_features=best_params["max_tfidf_features"])

    file_path = r"C:\Users\Tolgahan\Downloads\LinkedIn-personalities5.CSV"
    df = processor.load_data(file_path)
    X, y = processor.preprocess(df, fit_vectorizer=True)

    mbti_module = MBTIModule(
        model_name="bert-base-multilingual-cased",
        maxlen=best_params["max_length"],
        max_tfidf_features=best_params["max_tfidf_features"],
        epochs=5,  # Hier kann man die KI in größeren Iterationen trainieren lassen
        batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"],
        dense_units=best_params["dense_units"],
        dropout_rate=best_params["dropout_rate"]
    )

    mbti_module.train(X, y)
    mbti_module.save("mbti_module_best.keras")

    # Hier könnte auch der Predictor initialisiert und getestet werden,
    # wenn wir den tfidf-vectoriser speichern und beim Inferenzaufruf laden.
    print("Modell trainiert und gespeichert.")


if __name__ == "__main__":
    main()
