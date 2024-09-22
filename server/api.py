from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loguru import logger
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Ajouter le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permettre toutes les origines, remplace "*" par une liste d'origines spécifiques si nécessaire
    allow_credentials=True,
    allow_methods=["*"],  # Permettre toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Permettre tous les en-têtes
)



# Définition des modèles pour les données d'entrée
class PredictionData(BaseModel):
    surface: float

class PredictionDataAppartement(BaseModel):
    surface: float
    nbRooms: float
    nbWindows: float
    price: float

# Initialisation des modèles
model = LinearRegression()
modelSecond = LogisticRegression(max_iter=200)
modelThird = KNeighborsClassifier(n_neighbors=5)

label_encoder = LabelEncoder()
is_model_trained = False

@app.post("/train")
async def train():
    global is_model_trained

    # Lire le fichier CSV
    df = pd.read_csv('appartementsTraining.csv')

    # Extraction des variables pour le modèle de prix
    X_price = df[['surface']]
    y_price = df['price']

    # Entraînement du modèle de prix
    model.fit(X_price, y_price)

    # Catégorisation du prix
    bins = [0, 150000, 250000, 400000, float('inf')]
    labels = ['low', 'normal', 'high', 'scam']
    df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels)

    # Extraction des variables pour le modèle de classification
    X_category = df[['nbRooms', 'surface', 'nbWindows', 'price']]
    y_category = df['price_category']

    # Séparation des données
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_category, y_category, test_size=0.2, random_state=42)
    
    # Entraînement du modèle de catégorie
    modelSecond.fit(X_train_cat, y_train_cat)

    # Fonction pour classifier par type d'appartement
    def classify_apartment_by_surface(surface):
        if surface < 40:
            return 'F1'
        elif 40 <= surface < 60:
            return 'F2'
        elif 60 <= surface < 80:
            return 'F3'
        else:
            return 'F4'

    df['apartment_type'] = df['surface'].apply(classify_apartment_by_surface)
    df['apartment_type_encoded'] = label_encoder.fit_transform(df['apartment_type'])

    # Extraction pour le modèle KNN
    X_type = df[['surface']]
    y_type = df['apartment_type_encoded']

    # Séparation des données
    X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X_type, y_type, test_size=0.2, random_state=42)
    
    # Entraînement du modèle KNN
    modelThird.fit(X_train_type, y_train_type)

    # Marquer le modèle comme entraîné
    is_model_trained = True
    logger.info("Modèles entraînés avec succès.")

    return {"message": "Modèles entraînés avec succès."}

@app.post("/predict")
async def predict(data: PredictionData):
    if not is_model_trained:
        raise HTTPException(status_code=400, detail="Le modèle n'est pas encore entraîné.")

    X_new = np.array([[data.surface]])
    predicted_price = model.predict(X_new)[0]

    logger.info(f"Prédiction faite pour surface: {data.surface}, Prix prédit: {predicted_price}")
    return {"predicted_price": predicted_price}

@app.post("/predict-category")
async def predict_category(data: PredictionDataAppartement):
    if not is_model_trained:
        raise HTTPException(status_code=400, detail="Le modèle n'est pas encore entraîné.")

    X_new = np.array([[data.nbRooms, data.surface, data.nbWindows, data.price]])
    predicted_category = modelSecond.predict(X_new)[0]

    logger.info(f"Prédiction de la catégorie pour surface: {data.surface}, Catégorie prédite: {predicted_category}")
    return {"predicted_category": predicted_category}

@app.post("/predict-type")
async def predict_type(data: PredictionData):
    if not is_model_trained:
        raise HTTPException(status_code=400, detail="Le modèle n'est pas encore entraîné.")

    X_new = np.array([[data.surface]])
    predicted_type_encoded = modelThird.predict(X_new)[0]
    predicted_type = label_encoder.inverse_transform([predicted_type_encoded])[0]

    logger.info(f"Prédiction du type pour surface: {data.surface}, Type prédite: {predicted_type}")
    return {"predicted_type": predicted_type}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
