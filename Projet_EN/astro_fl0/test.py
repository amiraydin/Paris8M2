from datetime import datetime

# Heure 1
heure1 = datetime.strptime("2024-01-01 14:30:00", "%Y-%m-%d %H:%M:%S")
# Heure 2
heure2 = datetime.strptime("2024-01-01 10:15:00", "%Y-%m-%d %H:%M:%S")

# Calcul de la différence
difference = heure1 - heure2

# Résultats
print("Différence en heures, minutes et secondes:", difference)
print("Différence en secondes:", difference.total_seconds())
