# MLOps de Extremo a Extremo: Detección de Fraude con Databricks y MLflow

Este proyecto implementa un pipeline completo de MLOps para detectar transacciones fraudulentas en un conjunto de datos altamente desbalanceado (0.17% de fraude). El flujo de trabajo está construido en la plataforma **Databricks (Serverless)**, utilizando **Apache Spark** para el procesamiento distribuido y **MLflow** para un seguimiento riguroso de los experimentos y la gestión de modelos.

## 🚀 Resumen del Proyecto y Objetivo de Negocio

El objetivo no era simplemente construir *un* modelo, sino construir un *sistema reproducible* para comparar modelos. El desafío de negocio central no era la "Exactitud" (Accuracy), sino gestionar el **conflicto entre Precisión y Recall** (es decir, "¿cuántos fraudes capturamos?" vs. "¿a cuántos clientes inocentes molestamos?").

## 🛠 Stack Tecnológico y Habilidades Demostradas

Este proyecto demuestra competencia en un stack de MLOps moderno sobre un entorno cloud.

### Stack Principal
* **Plataforma:** Databricks Free Edition (Serverless, Spark Connect)
* **Procesamiento de Datos:** Apache Spark (`pyspark.ml`)
* **Gestión de Datos:** Unity Catalog (Tablas y Volúmenes)
* **Manejo de Desbalance:** `imbalanced-learn` (SMOTE)
* **Seguimiento de Experimentos:** MLflow (integrado)
* **Modelado:** Spark ML (Logistic Regression, Random Forest)

### Habilidades Clave
* **Plataforma Databricks:** Configuración y uso de la plataforma Databricks para el desarrollo de proyectos de ML, incluyendo la gestión de cómputo *serverless* y la arquitectura *Spark Connect*.
* **MLflow:** Implementación del ciclo de vida de MLOps para el **seguimiento sistemático de experimentos**, incluyendo el registro de parámetros, métricas (AUC-PR, Recall, Precision) y artefactos de modelos.
* **Prevención de Fuga de Datos (Data Leakage):** Aplicación de una metodología de validación robusta, asegurando la división estratificada de datos *antes* de aplicar técnicas de sobremuestreo.
* **Manejo de Desbalance de Clases:** Implementación de SMOTE para corregir un desbalance extremo (0.17%), aplicándolo exclusivamente al conjunto de entrenamiento.
* **Análisis Costo-Beneficio:** Traducción de métricas técnicas (Precisión vs. Recall) en un análisis de impacto de negocio para justificar la selección final del modelo.

---

## Dataset del Proyecto

El conjunto de datos utilizado es el [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) de Kaggle, un estándar de la industria para problemas de detección de fraude.

Contiene transacciones de tarjetas de crédito realizadas en septiembre de 2013 por titulares de tarjetas europeos. Por razones de confidencialidad, las características originales han sido anonimizadas y transformadas mediante PCA (Análisis de Componentes Principales), resultando en 28 características (`V1` a `V28`). Las únicas características no transformadas son `Time` y `Amount`.

La característica más importante del dataset es su **desbalance de clases extremo**:
* **Total de Transacciones:** 284,807
* **Transacciones de Fraude (Clase 1):** 492
* **Porcentaje de Fraude: 0.172%**

Este severo desbalance justifica el uso de métricas como **AUC-PR** y **Recall** sobre la exactitud (Accuracy) y es el principal desafío metodológico del proyecto.

---

## 🔬 Metodología de MLOps

Este proyecto se centró en el rigor metodológico para garantizar resultados válidos.

### 1. Prevención de Fuga de Datos (Data Leakage)
El paso más crítico fue la gestión del desbalance. Se aplicó una **"Regla de Oro"**:

1.  El conjunto de datos completo (desbalanceado) se dividió primero en `train_data` (80%) y `test_data` (20%) usando división estratificada.
2.  El `test_data` se bloqueó y se mantuvo en su estado desbalanceado (0.17% fraude) para una evaluación honesta y realista.
3.  La técnica de sobremuestreo **SMOTE** se aplicó *exclusivamente* al `train_data`, creando un `train_balanced_data` para el entrenamiento.

Esto aseguró que el modelo nunca fuera evaluado con datos sintéticos, previniendo puntuaciones infladas y optimismo irreal.

### 2. Seguimiento Sistemático con MLflow
Cada experimento de modelo se envolvió en una ejecución `mlflow.start_run()`, registrando sistemáticamente:
* **Parámetros (`mlflow.log_param`):** Hiperparámetros del modelo (ej. `regParam`, `numTrees`).
* **Métricas (`mlflow.log_metric`):** Métricas de negocio clave (AUC-PR, Precision, Recall) calculadas *solo* en el `test_data` desbalanceado.
* **Modelos (`mlflow.spark.log_model`):** Los artefactos del modelo entrenado, listos para ser desplegados.

---

## 📊 Resultados y Selección Final del Modelo

El uso de MLflow nos permitió comparar los modelos en un único cuadro de mando, centrándonos en el equilibrio Precisión-Recall.

| Modelo | AUC-PR | **Recall** (Captura de Fraude) | **Precision** (Exactitud de Alerta) |
| :--- | :---: | :---: | :---: |
| `Logistic Regression` | 0.632 | 0.817 (81.7%) | **0.401 (40.1%)** |
| `Random Forest` | 0.693 | **0.854 (85.4%)** | 0.263 (26.3%) |

### Conclusión y Decisión de Negocio

Se identificó un claro dilema de negocio:

* El **Random Forest** capturó más fraude (Recall del 85.4%), pero a un costo inaceptable: su Precisión se desplomó al 26.3%. Esto significaría que ~74 de cada 100 alertas de fraude serían falsos positivos, inundando al equipo de operaciones y creando una pésima experiencia para el cliente.
* La **Regresión Logística** ofreció un equilibrio mucho más eficiente. Aunque sacrificó un ~4% de captura de fraude, su Precisión (40.1%) fue significativamente mejor.

**Decisión Final:** Se seleccionó el modelo **`Logistic Regression`** como el candidato para producción. Proporciona un fuerte `Recall` mientras mantiene los falsos positivos en un nivel manejable, alineándose con el objetivo de negocio de equilibrar la detección de riesgos con la satisfacción del cliente.

---

## Imágenes
### Resultados del experimento
<img width="1031" height="248" alt="image" src="https://github.com/user-attachments/assets/5cec5447-81b1-44e0-a0b6-2f53b4c5572f" />

### Databricks Catalog
<img width="222" height="299" alt="image" src="https://github.com/user-attachments/assets/ddd68ba6-2a97-48fa-996c-0565f53adaf4" />

