import numpy as np
import matplotlib.pyplot as plt

def plot_temperatures(days, temperatures):
    """Trace les températures pour chaque jour de la semaine."""
    plt.figure(figsize=(10, 6))
    plt.plot(days, temperatures, marker='o', linestyle='-', color='b')
    plt.title("Températures Journalières")
    plt.xlabel("Jour")
    plt.ylabel("Température (°C)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_normalized_temperatures(days, normalized_temperatures):
    """Trace les températures normalisées pour chaque jour."""
    plt.figure(figsize=(10, 6))
    plt.bar(days, normalized_temperatures, color='skyblue')
    plt.title("Températures Normalisées")
    plt.xlabel("Jour")
    plt.ylabel("Valeurs Normalisées")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_predictions(X_train, y_train, X_test, y_test, predictions, model):
    """Trace les prédictions du modèle de régression linéaire."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='Données d\'entraînement')
    plt.scatter(X_test, y_test, color='green', label='Données de test')
    plt.scatter(X_test, predictions, color='red', label='Prédictions')
    
    X_combined = np.vstack((X_train, X_test))
    y_pred = model.predict(X_combined)
    plt.plot(X_combined, y_pred, color='black', linestyle='--', label='Ligne de Régression')
    
    plt.title("Régression Linéaire - Températures")
    plt.xlabel("Jour (encodé)")
    plt.ylabel("Température (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()