import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut, LeavePOut, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv("irisbin.csv")

# Dividir los datos en características (X) y etiquetas (y)
X = data.iloc[:, :-3].values
y = data.iloc[:, -3:].values

# Visualizar la proyección en dos dimensiones de la distribución de clases
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Codificar las etiquetas y en un formato adecuado para plt.scatter()
classes = np.unique(y)
colors = ['red', 'green', 'blue']
class_to_color = {classes[i]: colors[i] for i in range(len(classes))}
colors_mapped = [class_to_color[label[0]] for label in y]

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_mapped)
plt.title('Proyección en 2D de las clases del dataset Iris')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.grid(True)

# Mostrar el gráfico
plt.show()

# Dividir los datos en conjuntos de entrenamiento y prueba (80% para entrenamiento, 20% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de perceptrón multicapa
model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Método leave-one-out
loo = LeaveOneOut()
loo_scores = cross_val_score(model, X, y, cv=loo)
loo_error = 1 - np.mean(loo_scores)
loo_std = np.std(loo_scores)

# Método leave-k-out (con k=5)
lpo = LeavePOut(p=5)
lpo_scores = cross_val_score(model, X, y, cv=lpo)
lpo_error = 1 - np.mean(lpo_scores)
lpo_std = np.std(lpo_scores)

# Mostrar resultados en la consola después de una pausa
input("Presiona Enter para mostrar los resultados en la consola")

print(f"Error de clasificación (leave-one-out): {loo_error:.2f}")
print(f"Desviación estándar (leave-one-out): {loo_std:.2f}")
print(f"Error de clasificación (leave-k-out, k=5): {lpo_error:.2f}")
print(f"Desviación estándar (leave-k-out, k=5): {lpo_std:.2f}")