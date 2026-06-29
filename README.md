# World Happiness Report — Práctica Final Estadística
**Autor:** Iván Wilfrido Delgado Chaparro  
**Módulo:** Estadística para Data Science  
**Máster:** Data Science & IA — Evolve Academy  
**Dataset:** World Happiness Report (2015–2019) — Kaggle  

---

## Dataset

| Característica | Valor |
|---|---|
| Fuente | Kaggle — World Happiness Report |
| Archivos originales | 5 CSVs (uno por año) |
| Filas tras unión | 782 |
| Países | 170 |
| Rango temporal | 2015 – 2019 |
| Variable objetivo | Happiness_Score (continua, rango 2.69–7.77) |

**Columnas principales:** Country, Happiness_Score, GDP_per_Capita, Social_Support, Life_Expectancy, Freedom, Generosity, Corruption, Region, Happiness_Level, Year

---

## Estructura del proyecto

practica_final_delgado_chaparro_ivanwilfrido/
│
├── ejercicio1_descriptivo.py
├── ejercicio2_inferencia.py
├── ejercicio3_regresion_multiple.py
├── ejercicio4_series_temporales.py
├── Respuestas.md
│
├── data/
│   └── happiness_2015_2019.csv
│
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

---

## Preparación de datos

Los 5 CSVs originales tenían nombres de columnas distintos entre años y la variable Region solo existía en 2015 y 2016. El script `preparar_dataset.py` (fuera del proyecto entregable) se encargó de:

- Unir los 5 archivos añadiendo columna Year
- Homogeneizar nombres de columnas entre años
- Construir el mapa Country → Region desde 2015/2016
- Añadir Happiness_Level como segunda variable categórica (Low/Medium/High)
- Imputar el único nulo (UAE 2018, Corruption) con la mediana

---

## Ejercicios

### Ejercicio 1 — Análisis Estadístico Descriptivo
Análisis completo sobre el dataset: distribuciones con KDE, detección de outliers con método IQR, análisis de frecuencias de variables categóricas y mapa de calor de correlaciones de Pearson.

**Hallazgos clave:**
- GDP_per_Capita correlaciona 0.79 con Happiness_Score
- Corruption tiene 68 outliers (8.7%) — países con niveles extremos reales
- Sub-Saharan Africa representa el 25% del dataset con los scores más bajos
- Ningún par de variables supera multicolinealidad de 0.9

### Ejercicio 2 — Inferencia con Scikit-Learn
Regresión lineal para predecir Happiness_Score y regresión logística para clasificar países por nivel de felicidad.

| Métrica | Valor |
|---|---|
| MAE | 0.4395 |
| RMSE | 0.5728 |
| R² | 0.7303 |

Preprocesamiento: LabelEncoder para Region, StandardScaler, split 80/20 con random_state=42.

### Ejercicio 3 — Regresión Lineal Múltiple en NumPy
Implementación desde cero de OLS usando β = (XᵀX)⁻¹ Xᵀy con numpy.linalg.lstsq. Sin sklearn.

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

### Ejercicio 4 — Series Temporales
Descomposición aditiva con period=365, test ADF de estacionariedad y análisis del residuo.

| Estadístico | Valor |
|---|---|
| Media residuo | 0.1271 |
| Std residuo | 3.2220 |
| Jarque-Bera p-value | 0.5766 |
| ADF p-value | 0.0000 |

El residuo se comporta como ruido gaussiano: media ≈ 0, sin autocorrelación en ACF/PACF, normalidad no rechazada.

---

## Stack

Python · Pandas · NumPy · Scikit-Learn · Statsmodels · SciPy · Matplotlib · Seaborn · Kaggle
