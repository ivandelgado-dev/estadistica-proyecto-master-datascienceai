import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ─ Ejercicio 2 — Inferencia con Scikit-Learn ─────────────────────────────────

os.makedirs("output", exist_ok=True)

df = pd.read_csv("data/happiness_2015_2019.csv")

df["Happiness_Level"] = pd.Categorical(
    df["Happiness_Level"],
    categories=["Low", "Medium", "High"],
    ordered=True
)

print(df.shape)
print(df.dtypes)

# ── A) PREPROCESAMIENTO ──────────────────────────────────────────────────────

# eliminamos columnas que no aportan como predictoras
# Country es un identificador de texto, Year no es un predictor útil aquí
# Happiness_Level la dejamos fuera porque se deriva directamente del target
df_modelo = df.drop(columns=["Country", "Year", "Happiness_Level"])

# codificamos Region con LabelEncoder (la convierte a números)
le = LabelEncoder()
df_modelo["Region"] = le.fit_transform(df_modelo["Region"])

# separamos features (X) y variable objetivo (y)
X = df_modelo.drop(columns=["Happiness_Score"])
y = df_modelo["Happiness_Score"]

# dividimos en train (80%) y test (20%) con semilla fija para reproducibilidad
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# escalamos las features para que todas tengan la misma magnitud
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print(f"Features: {list(X.columns)}")

# ── B) REGRESIÓN LINEAL ──────────────────────────────────────────────────────

modelo = LinearRegression()
modelo.fit(X_train_scaled, y_train)

# predicciones sobre el test set
y_pred = modelo.predict(X_test_scaled)

# métricas de evaluación
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nMétricas de regresión lineal:")
print(f"  MAE  = {mae:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  R²   = {r2:.4f}")

# guardamos las métricas en output/
with open("output/ej2_metricas_regresion.txt", "w", encoding="utf-8") as f:
    f.write("=" * 45 + "\n")
    f.write(f"MAE  : {mae:.6f}\n")
    f.write(f"RMSE : {rmse:.6f}\n")
    f.write(f"R²   : {r2:.6f}\n")
print(" - ej2_metricas_regresion.txt guardado")

# coeficientes del modelo — nos dicen cuánto influye cada variable
print("\nCoeficientes del modelo:")
for nombre, coef in zip(X.columns, modelo.coef_):
    print(f"  {nombre}: {coef:.4f}")

# ── C) GRÁFICO DE RESIDUOS ───────────────────────────────────────────────────

# los residuos son la diferencia entre el valor real y el predicho
# si el modelo es bueno deben distribuirse aleatoriamente alrededor del 0
residuos = y_test - y_pred

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred, residuos, alpha=0.5, color="steelblue", edgecolors="white")
ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
ax.set_title("Gráfico de residuos — Regresión Lineal")
ax.set_xlabel("Valores predichos")
ax.set_ylabel("Residuos")
plt.tight_layout()
plt.savefig("output/ej2_residuos.png", dpi=150, bbox_inches="tight")
plt.close()
print(" - ej2_residuos.png guardado")

# ── D) GRÁFICO DE COEFICIENTES ───────────────────────────────────────────────

# visualizamos qué variables influyen más y en qué dirección
nombres_coefs = list(X.columns)
valores_coefs = modelo.coef_

fig, ax = plt.subplots(figsize=(8, 5))
colores = ["tomato" if c < 0 else "steelblue" for c in valores_coefs]
ax.barh(nombres_coefs, valores_coefs, color=colores, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Coeficientes del modelo — Regresión Lineal")
ax.set_xlabel("Valor del coeficiente")
plt.tight_layout()
plt.savefig("output/ej2_coeficientes.png", dpi=150, bbox_inches="tight")
plt.close()
print(" - ej2_coeficientes.png guardado")

# ── E) CLASIFICACIÓN Y MATRIZ DE CONFUSIÓN ───────────────────────────────────

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# usamos Happiness_Level como target (Low/Medium/High)
# la codificamos a número para el modelo
le_target = LabelEncoder()
y_cat = le_target.fit_transform(df["Happiness_Level"].astype(str))

X_cat = df_modelo.drop(columns=["Happiness_Score"])

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X_cat, y_cat, test_size=0.2, random_state=42
)

# escalamos igual que antes
X_train_cat_sc = scaler.fit_transform(X_train_cat)
X_test_cat_sc  = scaler.transform(X_test_cat)

# entrenamos regresión logística
modelo_log = LogisticRegression(random_state=42, max_iter=1000)
modelo_log.fit(X_train_cat_sc, y_train_cat)

y_pred_cat = modelo_log.predict(X_test_cat_sc)

# matriz de confusión
cm = confusion_matrix(y_test_cat, y_pred_cat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=le_target.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False)
ax.set_title("Matriz de confusión — Regresión Logística")
plt.tight_layout()
plt.savefig("output/ej2_matriz_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print(" - ej2_matriz_confusion.png guardado")