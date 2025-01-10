from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(X, y):
    """Entraîne un modèle de régression linéaire.
    
    Args:
        X (array-like): Features d'entraînement
        y (array-like): Variables cibles
        
    Returns:
        model: Modèle entraîné
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X et y doivent être des arrays numpy")
        
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    return model

def evaluate_model(model, X, y):
    """Évalue le modèle en calculant l'erreur quadratique moyenne.
    
    Args:
        model: Modèle entraîné
        X (array-like): Features de test
        y (array-like): Variables cibles de test
        
    Returns:
        tuple: (erreur quadratique moyenne, prédictions)
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X et y doivent être des arrays numpy")
        
    predictions = model.predict(X.reshape(-1, 1))
    mse = mean_squared_error(y, predictions)
    return mse, predictions