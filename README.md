# Actividad 5: predicción con datos de empleados
# Valeria Ortiz Velarde

Este proyecto contiene un notebook de Google Colab que descarga y analiza un conjunto de datos de empleados. El notebook entrena un modelo para clasificar si una persona deja la empresa y compara dos modelos para estimar sus años de experiencia en el dominio actual. También guarda las métricas, los modelos y las gráficas en MLflow.


## Archivos del proyecto


| Archivo | Para qué sirve |
|---|---|
| `fuentes/entrena.ipynb` | Notebook que ejecuta todo el análisis: descarga, limpieza, gráficas, entrenamiento, evaluación y registro en MLflow. |
| `Informe_Analisis_Modelado_MLflow.docx` | Informe Word que explica los datos y la metodología, y contiene tablas para documentar las métricas y la evidencia de MLflow. |
| `README.md` | Estas instrucciones. |


El notebook está diseñado para ejecutarse por sí mismo en Colab. No necesita archivos `.py` auxiliares.


## Qué modelos se entrenan


### Clasificación: ¿la persona deja la empresa?


- Columna que se quiere predecir: `LeaveOrNot`.
- El notebook asume que `1` significa que la persona deja la empresa y `0` que permanece.
- Modelo: regresión logística.
- Ajuste: busca los valores de `C` y `class_weight` que obtienen el mejor F1 en validación cruzada.
- Métricas finales: accuracy, precision, recall, F1 y ROC-AUC.


### Regresión: estimación de años de experiencia


- Columna que se quiere predecir: `ExperienceInCurrentDomain`.
- Modelos comparados: regresión lineal y XGBoost.
- Ajuste: selecciona los hiperparámetros que minimizan MAE en validación cruzada.
- Métricas finales: MAE, RMSE y R².


La regresión es una tarea de práctica: este dataset no incluye salario ni otra medida continua de negocio. `ExperienceInCurrentDomain` es un conteo discreto de años. El notebook excluye esa columna de las entradas del regresor para que el modelo no reciba directamente la respuesta que debe estimar.


## Cómo ejecutar el notebook en Google Colab


1. Abre Google Colab y carga `fuentes/entrena.ipynb`.
2. En Colab, selecciona **Entorno de ejecución → Ejecutar todas** (o ejecuta las celdas una por una desde arriba).
3. La primera celda instala las bibliotecas requeridas: KaggleHub, MLflow, scikit-learn, XGBoost, pandas, NumPy y Matplotlib.
4. Cuando se solicite, autoriza el acceso a Kaggle. KaggleHub descarga la versión más reciente del dataset `tawfikelmetwally/employee-dataset`.
5. Espera a que terminen la limpieza, la validación cruzada y el entrenamiento. Grid search prueba varios valores de hiperparámetros y puede tardar unos minutos.
6. Al final, revisa la tabla de validación de MLflow. Debe indicar que las corridas tienen parámetros, métricas, un modelo y gráficas.
7. Descarga el archivo `resultados_actividad5.zip` cuando Colab lo solicite.


Si Colab se desconecta o se reinicia, vuelve a abrir el notebook y ejecuta las celdas nuevamente. Los archivos guardados en `/content` son temporales y se pierden al terminar la sesión.


## Acceso a Kaggle


El dataset se descarga con esta instrucción:


```python
kagglehub.dataset_download("tawfikelmetwally/employee-dataset")
```


Si Kaggle solicita autenticación, configura tus credenciales de Kaggle en Colab y vuelve a ejecutar la celda de descarga. No subas el archivo de credenciales al repositorio ni lo incluyas en el notebook.


Fuente del dataset: [Employee Dataset en Kaggle](https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset).


## Qué hace el notebook, paso a paso


1. **Descarga:** obtiene el CSV y conserva una copia original.
2. **Explora:** muestra la distribución de `LeaveOrNot`, histogramas de edad y experiencia, y una matriz de correlaciones.
3. **Limpia:** quita espacios sobrantes, normaliza texto, convierte las columnas numéricas, elimina duplicados y descarta filas con valores faltantes en las variables usadas. Comprueba que `LeaveOrNot` tenga solo valores 0 y 1.
4. **Divide los datos:** reserva 20 % para la evaluación final y usa el 80 % restante para entrenar y ajustar los modelos. La clasificación conserva la proporción de clases.
5. **Prepara las columnas:** convierte categorías a indicadores (*one-hot encoding*) y estandariza variables numéricas dentro de los pipelines. Así, cada fold aprende el preprocesamiento solo con sus datos de entrenamiento.
6. **Ajusta y evalúa:** usa cinco folds de validación cruzada para seleccionar hiperparámetros. Calcula las métricas finales en el conjunto de prueba reservado.
7. **Grafica:** crea matriz de confusión y curva ROC para clasificación; para regresión, gráficos de valores observados frente a predichos, residuos y comparación de métricas.
8. **Registra:** guarda las corridas y sus artefactos en MLflow y comprueba que el registro haya terminado correctamente.


## Archivos que produce la ejecución


El notebook crea estas carpetas dentro de `/content/Actividad5`:


| Ruta | Contenido |
|---|---|
| `datos/datos_ini/employee_dataset.csv` | Copia descargada del CSV original. |
| `datos/datos_limp/employee_dataset_clean.csv` | Datos después de la limpieza. |
| `resultados/graficas/` | Imágenes de exploración y evaluación. |
| `resultados.json` | Mejores parámetros, métricas de validación cruzada y métricas de prueba. |
| `mlflow.db` | Base SQLite de MLflow cuando se usa seguimiento local. |
| `resultados_actividad5.zip` | Paquete descargable con el JSON, las gráficas y la base de MLflow local. |



## Cómo leer las métricas


- **Accuracy:** proporción de predicciones de clasificación correctas.
- **Precision:** de los casos que el modelo clasificó como salida, cuántos eran realmente salida.
- **Recall:** de los casos reales de salida, cuántos detectó el modelo.
- **F1:** combina precision y recall; es la métrica usada para elegir la regresión logística.
- **ROC-AUC:** mide qué tan bien ordena el modelo los casos positivos por encima de los negativos.
- **MAE:** error absoluto promedio en años; más bajo es mejor.
- **RMSE:** error en años que penaliza más los errores grandes; más bajo es mejor.
- **R²:** variación explicada por el regresor; normalmente más alto es mejor, pero puede ser negativo.


Compara los regresores usando el mismo conjunto de prueba. Una diferencia de MAE o R² por sí sola no demuestra que un modelo sea estadísticamente superior; el informe Word propone un análisis pareado que todavía debe calcularse después de la ejecución.


## MLflow: seguimiento local o remoto


### Usar el seguimiento local


Es la opción predeterminada. El notebook crea `mlflow.db` en `/content/Actividad5` y registra cuatro corridas: clasificación, regresión lineal, XGBoost y comparación de regresores. Las primeras tres corridas guardan cada modelo, sus parámetros, métricas y gráficas. La corrida de comparación guarda las métricas de ambos regresores y la gráfica comparativa; no corresponde a un modelo separado.


La celda de validación consulta MLflow y comprueba que las cuatro corridas tengan parámetros, métricas y gráficas, y que las tres corridas de modelos tengan además el archivo del modelo. Si todo está registrado, muestra una tabla de validación; si falta un elemento esperado, la celda informa un error.


Para abrir la interfaz en tu computadora, descarga y descomprime `resultados_actividad5.zip`, instala MLflow y ejecuta desde la carpeta que contiene `mlflow.db`:


```bash
pip install mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```


Después visita <http://127.0.0.1:5000>.


### Usar MLflow remoto


Necesitas un servidor MLflow accesible desde internet y desde Colab. En **Secrets** de Colab, configura `MLFLOW_TRACKING_URI` con la dirección del servidor. Si el servicio usa autenticación básica, configura también `MLFLOW_TRACKING_USERNAME` y `MLFLOW_TRACKING_PASSWORD`. Ejecuta la celda de configuración y después el resto del notebook. No escribas contraseñas ni tokens directamente en una celda.


La cuenta debe tener permiso para crear experimentos y corridas, guardar modelos y cargar artefactos. Si la celda de validación da un error, confirma la dirección, las credenciales, los permisos y que el servidor acepte conexiones desde Colab.


## Informe Word y estado de sus resultados


`Informe_Analisis_Modelado_MLflow.docx` documenta la fuente y las variables, las reglas de limpieza, el versionamiento, los modelos y sus hiperparámetros, las métricas, la comparación estadística, la evidencia de MLflow y los riesgos MLOps.


Las tablas de métricas y evidencia aparecen como **pendientes** hasta ejecutar el notebook. Después de la ejecución, usa `resultados.json` para completar las métricas y guarda la salida de validación o capturas de MLflow como evidencia. No reemplaces esos campos con cifras de ejemplo.




