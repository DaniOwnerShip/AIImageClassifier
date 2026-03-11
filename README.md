# Proyecto (2023) de Clasificación de Imágenes: Humano vs Perro

Este proyecto usa **TensorFlow/Keras** para entrenar una red neuronal convolucional (CNN) que clasifica imágenes entre humano y perro (o dog/food según tu dataset).  

---

## Archivos. Uso con entrenamiento de modelo

### 1. `fullTrainer.py` – Entrenamiento completo desde cero
- Lee los datos de entrenamiento y prueba.  
- Preprocesa las imágenes (redimensiona a 128x128 y convierte a escala de grises).  
- Define el modelo CNN y lo entrena usando los datos de prueba como validación.  
- Evalúa la precisión del modelo.  

**Uso**: 
python fullTrainer.py

**Salida esperada**:
- Precisión del modelo en el conjunto de prueba.  
- Visualización de algunas imágenes de entrenamiento con sus etiquetas.  

### 2. `trainerSaveModel.py` – Entrena y guarda el modelo
- Divide los datos en entrenamiento y validación (80% / 20%).  
- Preprocesa las imágenes igual que `full_trainer.py`.  
- Entrena la CNN y guarda el modelo entrenado en `trained_model-10-16.pkl`.  
- Permite usar TensorBoard para visualizar métricas de entrenamiento.  

**Uso**:
python trainerSaveModel.py

**Salida esperada**:
- Precisión en el conjunto de validación.  
- Archivo `trained_model-10-16.pkl` listo para usar en predicciones.  

### 3. `imagePredictor.py` – Predicción de nuevas imágenes
- Carga una imagen nueva (asegúrate de que sea 128x128 y en escala de grises).  
- Carga el modelo guardado `trained_model-10-16.pkl`.  
- Realiza la predicción y muestra la imagen con la etiqueta estimada.  

**Uso**:
python imagePredictor.py

**Salida esperada**:
- Valor de predicción (0 o 1).  
- Etiqueta asignada (`dog` o `food`).  
- Visualización de la imagen con el resultado.  

---

## Uso Sin entrenamiento de modelo

1. Coloca la imagen de prueba en la carpeta `./ImgTestHuman/`.  
2. Asegúrate de tener el modelo entrenado `trained_model-10-16.pkl` en la carpeta raíz del proyecto.  
3. Ejecuta el script: imagePredictor.py
4. Se mostrará:
   - Predicción en consola (`prediction`, `prediction_label` y `tag`)  
   - Imagen con visualización de la predicción para verificación humana



## Requisitos

- Python 3.8+  
- TensorFlow 2.x  
- Keras  
- Pandas  
- NumPy  
- Matplotlib  
- scikit-learn  
