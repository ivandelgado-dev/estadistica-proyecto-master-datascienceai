# World Happiness Report — Práctica Final de Estadística
Análisis estadístico completo sobre datos de felicidad por países, enmarcado en el módulo de Estadística para Data Science del Máster en Data Science & IA en Evolve Academy.

---

## Problema
Los informes anuales del World Happiness Report vienen en archivos separados con columnas inconsistentes entre años. El objetivo es unificarlos, limpiarlos y aplicar sobre ellos el ciclo completo de análisis estadístico: desde la descripción de los datos hasta la predicción y el análisis temporal.

---

## Datos
- **Fuente:** [World Happiness Report — Kaggle](https://www.kaggle.com/datasets/unsdsn/world-happiness)
- **Formato original:** 5 CSVs independientes (2015–2019), cada uno con nombres de columnas distintos
- **Dataset final:** 782 filas · 170 países · 11 columnas
- **Variable objetivo:** `Happiness_Score` (continua, rango 2.69–7.77)
- **Variables categóricas:** `Region` (10 regiones geográficas) y `Happiness_Level` (Low / Medium / High, construida a partir del score)

---

## Enfoque

### Preparación de datos
Los 5 CSVs tenían nombres de columna distintos entre años y la variable `Region` solo existía en 2015 y 2016. El script de preparación:
- Une los 5 archivos añadiendo columna `Year`
- Homogeneiza nombres de columnas entre años
- Construye el mapa `Country → Region` desde 2015/2016 y lo aplica a los años restantes con asignaciones manuales para los países nuevos
- Crea `Happiness_Level` como segunda variable categórica
- Imputa el único nulo del dataset (UAE 2018, columna `Corruption`) con la mediana

### Ejercicio 1 — Análisis descriptivo
Análisis completo con histogramas + KDE, detección de outliers por método IQR, frecuencias de variables categóricas y mapa de calor de correlaciones de Pearson.

Se eligió IQR sobre Z-score porque las variables `Generosity` y `Corruption` tienen distribuciones asimétricas y el Z-score asume normalidad.

### Ejercicio 2 — Inferencia con Scikit-Learn
Regresión lineal para predecir `Happiness_Score` y regresión logística para clasificar países por nivel de felicidad. Preprocesamiento: `LabelEncoder` para `Region`, `StandardScaler` para todas las features, split 80/20 con `random_state=42`.

### Ejercicio 3 — OLS desde cero con NumPy
Implementación de la solución analítica β = (XᵀX)⁻¹ Xᵀy usando `numpy.linalg.lstsq` en lugar de invertir la matriz directamente, por estabilidad numérica. Sin sklearn.

### Ejercicio 4 — Series temporales
Descomposición aditiva con `period=365`, test ADF de estacionariedad sobre el residuo, análisis ACF/PACF y test de normalidad Jarque-Bera.

---

## Resultados

### Ejercicio 1 — Análisis descriptivo
| Variable | Correlación con Happiness_Score |
|---|---|
| GDP_per_Capita | 0.789 |
| Life_Expectancy | 0.742 |
| Social_Support | 0.649 |

- 68 outliers en `Corruption` (8.7%) y 17 en `Generosity` (2.17%) — mantenidos por ser valores reales de países con características extremas
- Ningún par de variables supera multicolinealidad de 0.9
- `Sub-Saharan Africa` representa el 25% del dataset con los scores más bajos de forma consistente

### Ejercicio 2 — Regresión lineal
| Métrica | Valor |
|---|---|
| MAE | 0.4395 |
| RMSE | 0.5728 |
| R² | 0.7303 |

El modelo explica el 73% de la variabilidad del score. `GDP_per_Capita` es el predictor con mayor peso (coeficiente 0.44).

### Ejercicio 3 — OLS con NumPy
| Parámetro | Valor real | Valor ajustado |
|---|---|---|
| β₀ | 5.0 | 4.8650 |
| β₁ | 2.0 | 2.0636 |
| β₂ | -1.0 | -1.1170 |
| β₃ | 0.5 | 0.4385 |

| Métrica | Valor |
|---|---|
| MAE | 1.1665 |
| RMSE | 1.4612 |
| R² | 0.6897 |

### Ejercicio 4 — Series temporales
| Estadístico del residuo | Valor |
|---|---|
| Media | 0.1271 |
| Desviación típica | 3.2220 |
| Jarque-Bera p-value | 0.5766 |
| ADF p-value | 0.0000 |

El residuo se comporta como ruido gaussiano: media ≈ 0, normalidad no rechazada (p > 0.05), estacionario (ADF p < 0.05), sin autocorrelación visible en ACF/PACF.

---

## Stack
Python · Pandas · NumPy · Scikit-Learn · Statsmodels · SciPy · Matplotlib · Seaborn · Kaggle

---

## Estructura del repo

```
practica_final_delgado_chaparro_ivanwilfrido/
├── ejercicio1_descriptivo.py
├── ejercicio2_inferencia.py
├── ejercicio3_regresion_multiple.py
├── ejercicio4_series_temporales.py
├── Respuestas.md
├── data/
│   └── happiness_2015_2019.csv
└── output/
    ├── ej1_descriptivo.csv
    ├── ej1_outliers.txt
    ├── ej1_histogramas.png
    ├── ej1_boxplots.png
    ├── ej1_heatmap_correlacion.png
    ├── ej1_categoricas.png
    ├── ej2_metricas_regresion.txt
    ├── ej2_residuos.png
    ├── ej2_coeficientes.png
    ├── ej2_matriz_confusion.png
    ├── ej3_coeficientes.txt
    ├── ej3_metricas.txt
    ├── ej3_predicciones.png
    ├── ej4_serie_original.png
    ├── ej4_descomposicion.png
    ├── ej4_acf_pacf.png
    ├── ej4_histograma_ruido.png
    └── ej4_analisis.txt
```
