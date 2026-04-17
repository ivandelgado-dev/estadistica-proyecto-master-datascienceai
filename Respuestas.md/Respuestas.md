# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
El dataset es el World Happiness Report de Kaggle, que venía dividido en 5 archivos CSV, uno por año. Los uní, homogeneicé los nombres de columnas y añadí la variable Region tomándola de los años donde existía. El resultado es un dataset de 782 filas y 11 columnas listo para analizar.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset es el World Happiness Report de Kaggle, que recoge datos de felicidad por países entre 2015 y 2019. La variable objetivo es Happiness_Score. Tiene sentido hacer regresión porque es un número continuo que depende de otros factores como el PIB o la esperanza de vida, así que podemos intentar predecirlo.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> La mayoría de variables tienen una distribución bastante normal. Generosity y Corruption se salen un poco, tienen bastantes valores altos que se alejan del resto. Con el método IQR encontré 17 outliers en Generosity y 68 en Corruption. Los dejé porque son datos reales de países, no errores.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Las tres variables que más se relacionan con la felicidad son GDP_per_Capita (0.789), Life_Expectancy (0.742) y Social_Support (0.649). Tiene bastante sentido, a más riqueza y salud, más felicidad.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Solo había un valor nulo, en la columna Corruption para UAE en 2018, un 0.13% del total. Lo rellené con la mediana de esa columna.


---

## Ejercicio 2 — Inferencia con Scikit-Learn
---
Usando el dataset del ejercicio 1 entrené un modelo de regresión lineal con Scikit-Learn para predecir el Happiness_Score. También entrené una regresión logística para clasificar los países por nivel de felicidad y ver la matriz de confusión.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> El modelo da MAE=0.44, RMSE=0.57 y R²=0.73. No está mal, explica el 73% de la variación del score. No parece que haya overfitting. La variable que más pesa es GDP_per_Capita, lo cual tiene sentido viendo las correlaciones del ejercicio 1.Quité Country, Year y Happiness_Level porque no sirven como predictores. Region la convertí a número con LabelEncoder. Escalé todo con StandardScaler y dividí 80/20 con semilla 42.El heatmap del ejercicio 1 ya me decía qué variables iban a importar más, y el modelo lo confirmó. Al no haber multicolinealidad pude meter todas las variables sin problema. Si quisiera mejorar el modelo probaría algo no lineal. El gráfico de coeficientes confirma visualmente que GDP_per_Capita es la variable que más peso tiene, seguida de Life_Expectancy y Freedom. Region prácticamente no aporta.


---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy
---
Implementé desde cero la regresión lineal múltiple usando solo NumPy, sin sklearn. El modelo se prueba con datos sintéticos de coeficientes conocidos para comprobar que la implementación es correcta.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> La fórmula calcula los coeficientes que hacen que las predicciones se acerquen lo máximo posible a los valores reales. La columna de unos se añade para que el modelo tenga un punto de partida, el intercepto, que es el valor de y cuando todo lo demás es cero.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       |     4.8650     |
| β₁        | 2.0       |     2.0636     |
| β₂        | -1.0      |    -1.1170     |
| β₃        | 0.5       |     0.4385     |

> Los coeficientes que salieron están muy cerca de los reales, ninguno se aleja más de 0.2, que es el margen que da el profesor.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> MAE=1.1665, RMSE=1.4612 y R²=0.6897. Están dentro del rango que da el profesor como referencia, así que la implementación es correcta.

**Pregunta 3.4* — Compara los resultados con la reacción logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido. 

> La logística del ejercicio 2 clasifica bien los niveles de felicidad. Aquí el R² es algo menor que en sklearn porque en el ejercicio 2 escalé los datos y aquí no, al ser datos sintéticos.

---

## Ejercicio 4 — Series Temporales
---
Se trabaja con una serie temporal sintética de datos diarios entre 2018 y 2023. La analizamos visualmente, la descomponemos en sus partes y comprobamos si el residuo se parece a ruido gaussiano.


---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> Sí, hay una tendencia lineal que va subiendo. Sube unos 0.05 por día, en total unos 109 puntos en los 6 años.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Sí, hay estacionalidad anual, con un periodo de 365 días y una amplitud de unos 15 puntos entre el pico y el valle.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> También hay ciclos más largos, de unos 4 años. La diferencia con la tendencia es que estos suben y bajan, mientras que la tendencia solo sube.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> El residuo se comporta como ruido ideal. La media es casi 0, la std es 3.22, y tanto la asimetría como la curtosis están cerca de 0. El test Jarque-Bera da p=0.5766, así que no se rechaza la normalidad. El ADF da p=0.0000, el residuo es estacionario. En el ACF y PACF no se ve autocorrelación.

---

*Fin del documento de respuestas*
