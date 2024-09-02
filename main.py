import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD

# Lectura de los datos
data = pd.read_csv('altura_peso.csv')
x = data['Altura'].values
y = data['Peso'].values

# Normalizar los datos
x_mean = np.mean(x)
x_std = np.std(x)
x_normalized = (x - x_mean) / x_std

y_mean = np.mean(y)
y_std = np.std(y)
y_normalized = (y - y_mean) / y_std

# Crear el modelo
model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(1, activation='linear'))

# Definir el optimizador
optimizer = SGD(learning_rate=0.01)  # Ajustar la tasa de aprendizaje
model.compile(optimizer=optimizer, loss='mse')

# Entrenar el modelo
history = model.fit(x_normalized, y_normalized, epochs=10000, batch_size=len(x_normalized), verbose=0)

# Verificar la estructura del modelo
model.summary()

# Imprimir los parámetros del modelo
weights, bias = model.layers[0].get_weights()
print(f'Peso (w): {weights[0][0]}, Sesgo (b): {bias[0]}')

# Graficar el error cuadrático medio vs. el número de épocas
plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('Épocas')
plt.ylabel('Error Cuadrático Medio')
plt.title('ECM vs. Número de Épocas')
plt.show()

# Superponer la recta de regresión sobre los datos originales
plt.figure()
plt.scatter(x, y, label='Datos Originales')
plt.plot(x, model.predict((x - x_mean) / x_std) * y_std + y_mean, color='red', label='Recta de Regresión')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.legend()
plt.show()

# Predicción para una altura específica
altura_especifica = 170
altura_especifica_normalizada = (altura_especifica - x_mean) / x_std
peso_predicho_normalizado = model.predict(np.array([altura_especifica_normalizada]))
peso_predicho = peso_predicho_normalizado * y_std + y_mean
print(f'Predicción del peso para una altura de {altura_especifica} cm: {peso_predicho[0][0]} kg')