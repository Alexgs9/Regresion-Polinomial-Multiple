#Hecho por Alexandro Gutierrez Serna

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar el conjunto de datos
df = pd.read_csv("Fish.csv")

# Realizar label encoding para la columna "Species"
unique_species = df['Species'].unique()
species_mapping = {species: idx for idx, species in enumerate(unique_species)}
df['Species'] = df['Species'].map(species_mapping)

#print("dataframe")
#print(df)

# Variables independientes (X) y variable dependiente (y)
X = df[['Height', 'Width']].values
y = df['Weight'].values

# Agregar una columna de unos para el término independiente (intercept)
X = np.column_stack((np.ones(len(X)), X))

#print("X", X)

# Definir el grado del polinomio (puedes ajustarlo según tus necesidades)
grado = 2

# Crear características polinómicas
X_poly = X
for i in range(2, grado + 1):
    X_poly = np.column_stack((X_poly, X[:, 1:] ** i))

# Crear una nueva columna con la interacción entre 'Height' y 'Width'
X_poly = np.column_stack((X_poly, X[:, 1] * X[:, 2]))

#print("X_poly", X_poly)

# Dividir los datos en conjuntos de entrenamiento y prueba (80% y 20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)


# Calcular los coeficientes del modelo de regresión (usando la ecuación normal)
coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Imprimir los coeficientes
print("Coeficientes del modelo:")
print(coefficients)

#Prueba donde x1 = 11.52 y x2 = 4.02
#res = coefficients[0] + coefficients[1] * 11.52 + coefficients[2] * 4.02 + coefficients[3] * 11.52 ** 2 + coefficients[4] * 4.02 ** 2 + coefficients[5] * 11.52 * 4.02
#print("res: ", res)

#Ecuacion de polinomio multiple con x1 y x2
# Y = β0 + β1X1 + β2X2 + β3X1^2 + β4X2^2 + β5X1X2
print("Modelo: ", "Y = ", coefficients[0], " + (", coefficients[1], " * X1) + (", coefficients[2], " * X2) + (", coefficients[3], "* X1^2) + (", coefficients[4], " * X2^2) + (", coefficients[5], " * X1 * X2)")


# Función para hacer predicciones
def predict(X, coefficients):
    return X @ coefficients

predictions = predict(X_test, coefficients)
print("Predicciones de Weight")
print(predictions)
print("Valores reales de Weight")
print(y_test)

# Graficar las predicciones en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 1], X_test[:, 2], y_test, c='r', marker='o', label='Datos reales')
ax.scatter(X_test[:, 1], X_test[:, 2], predictions, c='b', marker='x', label='Predicciones')
ax.set_xlabel('Height')
ax.set_ylabel('Width')
ax.set_zlabel('Weight')
plt.legend()
plt.show()