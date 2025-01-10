import numpy as np

def load_data(file_path):
    """Charge les données à partir d'un fichier CSV."""
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=1)
    return data

def calculate_statistics(data):
    """Calcule la moyenne, le maximum et le minimum des données."""
    mean_temp = np.mean(data)
    max_temp = np.max(data)
    min_temp = np.min(data)
    return mean_temp, max_temp, min_temp

def normalize_data(data):
    """Normalise les données entre 0 et 1."""
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized


def encode_days(days):
    """Encode les jours de la semaine en indices numériques."""
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                   "Friday": 4, "Saturday": 5, "Sunday": 6}
    return np.array([day_mapping[day] for day in days])

