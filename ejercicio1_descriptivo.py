import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# ─ Ejercicio 1 — Análisis Estadístico Descriptivo ────────────────────────────

os.makedirs("output", exist_ok=True)

df = pd.read_csv("data/happiness_2015_2019.csv")

# Happiness_Level es categórica ordenada, se lo indicamos a pandas
df["Happiness_Level"] = pd.Categorical(
    df["Happiness_Level"],
    categories=["Low", "Medium", "High"],
    ordered=True
)

# ── A) RESUMEN ESTRUCTURAL ───────────────────────────────────────────────────
print(df.shape)
print(df.dtypes)
print(df.head())

# columnas numéricas que vamos a analizar (excluimos Year porque es un identificador)
cols_numericas = ["Happiness_Score", "GDP_per_Capita", "Social_Support",
                  "Life_Expectancy", "Freedom", "Generosity", "Corruption"]

# resumen estructural del dataset
print(f"\nFilas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
print(f"Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# porcentaje de nulos por columna
nulos = df.isnull().sum()
nulos_pct = (nulos / len(df) * 100).round(2)
print("\nValores nulos (%):")
print(nulos_pct)

# ── B) ESTADÍSTICOS DESCRIPTIVOS ────────────────────────────────────────────
# estadísticos descriptivos de las variables numéricas
descriptivos = df[cols_numericas].describe().T
descriptivos["variance"] = df[cols_numericas].var()
descriptivos["skewness"] = df[cols_numericas].skew()
descriptivos["kurtosis"] = df[cols_numericas].kurt()
descriptivos["IQR"] = descriptivos["75%"] - descriptivos["25%"]

print("\nEstadísticos descriptivos:")
print(descriptivos)

# guardamos la tabla en output/
descriptivos.to_csv("output/ej1_descriptivo.csv")
print("\n - ej1_descriptivo.csv guardado")

# ── C) DISTRIBUCIONES Y OUTLIERS ────────────────────────────────────────────
# detección de outliers con método IQR
# un valor es outlier si está por debajo de Q1 - 1.5*IQR o por encima de Q3 + 1.5*IQR
print("\nOutliers detectados por variable (método IQR):")
resumen_outliers = []

for col in cols_numericas:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]
    resumen_outliers.append({
        "variable": col,
        "n_outliers": len(outliers),
        "pct_outliers": round(len(outliers) / len(df) * 100, 2),
        "limite_inferior": round(limite_inferior, 4),
        "limite_superior": round(limite_superior, 4)
    })
    print(f"  {col}: {len(outliers)} outliers ({round(len(outliers)/len(df)*100,2)}%)")

# guardamos el resumen de outliers en output/
df_outliers = pd.DataFrame(resumen_outliers)
df_outliers.to_csv("output/ej1_outliers.txt", index=False)
print("\n - ej1_outliers.txt guardado")

# ── C.1) Histogramas ─────────────────────────────────────────────────────────
# histogramas de todas las variables numéricas con curva de densidad (KDE)
fig, ejes = plt.subplots(3, 3, figsize=(15, 10))
ejes = ejes.flatten()

for i, col in enumerate(cols_numericas):
    ejes[i].hist(df[col], bins=30, edgecolor="white", color="steelblue", alpha=0.7)
    # curva KDE encima del histograma
    kde_x = np.linspace(df[col].min(), df[col].max(), 200)
    kde = stats.gaussian_kde(df[col].dropna())
    eje_derecho = ejes[i].twinx()
    eje_derecho.plot(kde_x, kde(kde_x), color="darkorange", linewidth=2)
    eje_derecho.set_yticks([])
    ejes[i].set_title(col)
    ejes[i].set_xlabel("Valor")
    ejes[i].set_ylabel("Frecuencia")

# ocultamos el subplot sobrante (tenemos 7 variables y 9 celdas)
for j in range(len(cols_numericas), len(ejes)):
    ejes[j].set_visible(False)

plt.suptitle("Distribución de variables numéricas", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("output/ej1_histogramas.png", dpi=150, bbox_inches="tight")
plt.close()
print(" - ej1_histogramas.png guardado")

# ── C.2) Boxplots ────────────────────────────────────────────────────────────
# boxplots de la variable objetivo separados por cada variable categórica
fig, (eje1, eje2) = plt.subplots(1, 2, figsize=(14, 5))

# por región geográfica — ordenamos manualmente por mediana
regiones_orden = df.groupby("Region")["Happiness_Score"].median().sort_values().index.tolist()
datos_region = [df[df["Region"] == r]["Happiness_Score"].values for r in regiones_orden]
eje1.boxplot(datos_region, tick_labels=regiones_orden)
eje1.set_title("Happiness Score por Región")
eje1.set_xlabel("Región")
eje1.set_ylabel("Happiness Score")
eje1.tick_params(axis="x", rotation=45)

# por nivel de felicidad
niveles_orden = ["Low", "Medium", "High"]
datos_nivel = [df[df["Happiness_Level"] == n]["Happiness_Score"].values for n in niveles_orden]
eje2.boxplot(datos_nivel, tick_labels=niveles_orden)
eje2.set_title("Happiness Score por Nivel")
eje2.set_xlabel("Happiness Level")
eje2.set_ylabel("Happiness Score")

plt.suptitle("")
plt.tight_layout()
plt.savefig("output/ej1_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print(" - ej1_boxplots.png guardado")

# ── D) VARIABLES CATEGÓRICAS ────────────────────────────────────────────────
# frecuencias y gráficos de las variables categóricas
fig, (eje1, eje2) = plt.subplots(1, 2, figsize=(14, 5))

# frecuencias de Region
freq_region = df["Region"].value_counts()
freq_region_pct = (freq_region / len(df) * 100).round(2)
print("\nFrecuencias Region:")
print(pd.DataFrame({"absoluta": freq_region, "relativa (%)": freq_region_pct}))

freq_region.plot(kind="bar", ax=eje1, color="steelblue", edgecolor="white")
eje1.set_title("Frecuencia por Región")
eje1.set_xlabel("Región")
eje1.set_ylabel("Número de registros")
eje1.tick_params(axis="x", rotation=45)

# frecuencias de Happiness_Level
freq_nivel = df["Happiness_Level"].value_counts().reindex(["Low", "Medium", "High"])
freq_nivel_pct = (freq_nivel / len(df) * 100).round(2)
print("\nFrecuencias Happiness_Level:")
print(pd.DataFrame({"absoluta": freq_nivel, "relativa (%)": freq_nivel_pct}))

freq_nivel.plot(kind="bar", ax=eje2, color=["tomato", "gold", "mediumseagreen"], edgecolor="white")
eje2.set_title("Frecuencia por Happiness Level")
eje2.set_xlabel("Nivel")
eje2.set_ylabel("Número de registros")
eje2.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("output/ej1_categoricas.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n - ej1_categoricas.png guardado")

# análisis de desbalance
cat_dominante_region = freq_region_pct.max()
cat_dominante_nivel = freq_nivel_pct.max()
print(f"\nDesbalance Region: categoría más frecuente representa el {cat_dominante_region}%")
print(f"Desbalance Happiness_Level: categoría más frecuente representa el {cat_dominante_nivel}%")

# ── E) CORRELACIONES ────────────────────────────────────────────────────────
# mapa de calor de correlaciones entre variables numéricas
matriz_corr = df[cols_numericas].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    matriz_corr,
    annot=True,        # muestra los valores dentro de cada celda
    fmt=".2f",
    cmap="coolwarm",   # rojo = correlación positiva, azul = negativa
    center=0,
    square=True,
    ax=ax
)
ax.set_title("Matriz de correlaciones de Pearson")
plt.tight_layout()
plt.savefig("output/ej1_heatmap_correlacion.png", dpi=150, bbox_inches="tight")
plt.close()
print(" - ej1_heatmap_correlacion.png guardado")

# top 3 variables más correlacionadas con el target
corr_target = matriz_corr["Happiness_Score"].drop("Happiness_Score").abs().sort_values(ascending=False)
print("\nTop 3 correlaciones con Happiness_Score:")
print(corr_target.head(3))

# pares con posible multicolinealidad (|r| > 0.9, excluyendo diagonal)
print("\nPares con multicolinealidad potencial (|r| > 0.9):")
encontrado = False
for i in range(len(matriz_corr.columns)):
    for j in range(i+1, len(matriz_corr.columns)):
        val = abs(matriz_corr.iloc[i, j])
        if val > 0.9:
            print(f"  {matriz_corr.columns[i]} — {matriz_corr.columns[j]}: {val:.3f}")
            encontrado = True
if not encontrado:
    print("  Ninguno encontrado")
