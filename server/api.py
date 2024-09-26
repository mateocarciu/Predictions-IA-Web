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
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_squared_error, r2_score

app = FastAPI()

# cors middleware sinon on ne peut pas faire des requêtes depuis un autre domaine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionData(BaseModel):
    surface: float

class PredictionDataAppartement(BaseModel):
    surface: float
    nbRooms: float
    nbWindows: float
    price: float
    city: str

# Modèles
model = LinearRegression()
modelSecond = LogisticRegression(max_iter=200)
modelThird = KNeighborsClassifier(n_neighbors=5)
model_note = None
model_year = None
model_garage = None

label_encoder_city = LabelEncoder()
label_encoder_type = LabelEncoder()

is_model_trained = False

@app.post("/train")
async def train():
    global is_model_trained, model_note, model_year, model_garage

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

    # Encoder les types d'appartement
    df['apartment_type_encoded'] = label_encoder_type.fit_transform(df['apartment_type'])

    # Extraction pour le modèle KNN
    X_type = df[['surface']]
    y_type = df['apartment_type_encoded']

    # Séparation des données
    X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X_type, y_type, test_size=0.2, random_state=42)
    
    # Entraînement du modèle KNN
    modelThird.fit(X_train_type, y_train_type)

    # Encoder les villes
    df['city_encoded'] = label_encoder_city.fit_transform(df['city'])

    # Entraîner le modèle de note
    X_note = df[['city_encoded', 'surface', 'price']]
    y_note = df['note']
    
    model_note = LinearRegression()
    model_note.fit(X_note, y_note)

    # Entraîner le modèle d'année
    X_year = df[['city_encoded', 'price']]
    y_year = df['year']
    
    model_year = LinearRegression()
    model_year.fit(X_year, y_year)

    # Entraîner le modèle de garage
    y_garage = df['garage']
    model_garage = LogisticRegression(max_iter=200)
    model_garage.fit(X_year, y_garage)

    # Marquer le modèle comme entraîné
    is_model_trained = True
    logger.info("Tous les modèles ont été entraînés avec succès.")

    return {"message": "Tous les modèles ont été entraînés avec succès."}

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
    predicted_type = label_encoder_type.inverse_transform([predicted_type_encoded])[0]

    logger.info(f"Prédiction du type pour surface: {data.surface}, Type prédite: {predicted_type}")
    return {"predicted_type": predicted_type}

@app.post("/predict-note")
async def predict_note(data: PredictionDataAppartement):
    if model_note is None:
        raise HTTPException(status_code=400, detail="Le modèle de note n'est pas encore entraîné.")

    city_encoded = label_encoder_city.transform([data.city])[0]
    X_new = np.array([[city_encoded, data.surface, data.price]])
    predicted_note = model_note.predict(X_new)[0]

    return {"predicted_note": predicted_note}

@app.post("/predict-year")
async def predict_year(data: PredictionDataAppartement):
    if model_year is None:
        raise HTTPException(status_code=400, detail="Le modèle d'année n'est pas encore entraîné.")

    city_encoded = label_encoder_city.transform([data.city])[0]
    X_new = np.array([[city_encoded, data.price]])
    predicted_year = model_year.predict(X_new)[0]

    return {"predicted_year": predicted_year}

@app.post("/predict-garage")
async def predict_garage(data: PredictionDataAppartement):
    if model_garage is None:
        raise HTTPException(status_code=400, detail="Le modèle de garage n'est pas encore entraîné.")

    city_encoded = label_encoder_city.transform([data.city])[0]
    X_new = np.array([[city_encoded, data.price]])
    predicted_garage = model_garage.predict(X_new)[0]

    return {"predicted_garage": bool(predicted_garage)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
