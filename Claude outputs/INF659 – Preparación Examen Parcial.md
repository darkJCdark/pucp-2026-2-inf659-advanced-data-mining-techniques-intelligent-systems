# INF659 – Preparación para el Examen Parcial

Elaborado el 29 de septiembre de 2026 a partir de la carpeta del curso, el sílabo, la guía de estudio del Proyecto y el examen de práctica 2026-1.

## 1. Qué esperar

- **Cuándo:** semana 8 según el sílabo. La fecha exacta no figura en la carpeta; confírmala con los profesores (César Beltrán / Rodrigo Uribe).
- **Peso:** EP = 30 % de la nota final (NF = 0.2·TA + 0.3·EP + 0.5·PC).
- **Formato probable (inferido del examen 2026-1, no confirmado):** cuaderno Colab en TensorFlow/Keras con preguntas teóricas de justificación intercaladas. Total 18 pts: P1 = 2, P2 = 10, P3 = 2, P4 = 2, P4bis = 2, P5 = 2 (el notebook repite la numeración "Pregunta 4").
- **Temas evaluables (Unidades 1–5, S1–S6):** regresión y gradiente descendente, regresión logística/softmax, backprop, activaciones, inicialización, regularización (L2, BN, dropout), optimizadores, CNN, arquitecturas clásicas, transfer learning, metodología práctica.
- **IA generativa:** el sílabo indica que su uso es limitado o no permitido según cada evaluación; confirma la regla para el examen antes de usarla.

## 2. Lo que pidió el examen de práctica (2026-1)

Caso: clasificar CT scans 3D de pulmón (normal = 0 / anormal = 1), 100 volúmenes por clase (dataset tipo MosMedData), tamaño 128×128×64, HU normalizado de [-1000, 400] a [0, 1], split estratificado 85/15, `batch_size=2`, Adam con `ExponentialDecay`, `ModelCheckpoint` + `EarlyStopping` sobre `val_acc`.

| Pregunta | Pts | Qué había que hacer |
| --- | --- | --- |
| P1 | 2 | Analizar el data augmentation (rotación aleatoria ±5/10/20°): beneficio/perjuicio, diferencia sin él, y si agrega volúmenes nuevos o solo transforma. |
| P2 | 10 | Modelo base (conv + dense) → 2 pts; versión final según el paper → 5 pts; versión solo convolucional con GAP → 3 pts. Registrar configuraciones probadas. |
| P3 | 2 | Elegir parámetros de ExponentialDecay y explicar el efecto del learning rate. |
| P4 | 2 | Elegir `epochs` y `patience` y explicar su efecto. |
| P4bis | 2 | Cómo funciona una convolución 3D vs 2D (filtros); si transfer learning mejoraría y qué características debería tener el modelo origen. |
| P5 | 2 | Por qué accuracy es adecuada aquí y diferencia con F1, considerando dataset, cantidad y distribución de etiquetas. |

Además: identificar la mejor época y cargar ese checkpoint (`load_model`), y `classification_report`.

## 3. Respuestas modelo (para practicar redacción)

**P1 – Data augmentation.** La rotación aleatoria solo transforma los volúmenes existentes; no agrega volúmenes al corpus, sino variantes distintas en cada época (`tf.numpy_function` dentro de `map`). Beneficia: reduce el sobreajuste con solo ~170 volúmenes de entrenamiento y aporta invarianza a pequeñas rotaciones del paciente/equipo. Perjudica si el ángulo es excesivo (anatomía irreal) y añade costo de CPU. Sin augmentation se espera mayor brecha entre train y val. Se aplica solo a train (`test_augmentation` únicamente agrega el canal).

**P3 – ExponentialDecay.** `lr = lr0 · decay_rate^(step/decay_steps)`; con `staircase=True` decae en escalones. Un lr inicial de 1e-4 con Adam es prudente para datos pequeños. Si `decay_steps` es muy grande frente a los pasos reales (pocos: ~85 pasos/época con batch 2), el lr casi no decae y el schedule no cumple su función. Un decay más agresivo permite afinar cerca del mínimo pero puede estancar el entrenamiento.

**P4 – epochs y patience.** `epochs` es el tope de pasadas por los datos; `patience` es cuántas épocas sin mejora de `val_acc` se toleran antes de detener. Patience muy bajo corta antes de salir de mesetas iniciales (por eso el examen sugiere esperar ~10 épocas); muy alto gasta cómputo y sobreajusta. `ModelCheckpoint` guarda cada época, así que se puede recuperar la mejor aunque el entrenamiento siga después.

**P4bis – Conv 3D vs 2D.** El filtro es k×k×k×C_in y se desliza en las tres dimensiones (ancho, alto, profundidad), produciendo un mapa 3D por filtro; captura contexto entre cortes. Cuesta más parámetros y memoria (por eso batch 2). Transfer learning: ayudaría si el modelo origen se preentrenó en volúmenes médicos 3D (o imágenes médicas 2D adaptadas); un modelo de ImageNet 2D requiere reformular la entrada (canal y profundidad) y sufre desajuste de dominio.

**P5 – Accuracy vs F1.** Con 100 normales y 100 anormales el dataset está balanceado y el split es estratificado, por eso accuracy es una métrica razonable. F1 (media armónica de precision y recall) se vuelve indispensable con clases desbalanceadas o cuando falsos negativos y falsos positivos tienen costos distintos (en diagnóstico médico, el recall de la clase anormal es crítico). En un test de ~30 volúmenes, además, cada error mueve la accuracy ~3 puntos: hay que mencionarlo.

## 4. Arquitectura de referencia (P2)

El examen apunta al paper arXiv:2007.13224 (Zunair et al., uniformización de CT scans para 3D CNN; ImageCLEF Tuberculosis 2019) y al repositorio `hasibzunair/uniformizing-3D`. No pude acceder a la página 6 del paper; el diseño siguiente es el del ejemplo oficial de Keras "3D image classification from CT scans", que se basa en ese trabajo (verifícalo contra el paper):

```python
inputs = Input(shape=(128, 128, 64, 1))
x = Conv3D(64, 3, activation="relu")(inputs); x = MaxPool3D(2)(x); x = BatchNormalization()(x)
x = Conv3D(64, 3, activation="relu")(x);      x = MaxPool3D(2)(x); x = BatchNormalization()(x)
x = Conv3D(128, 3, activation="relu")(x);     x = MaxPool3D(2)(x); x = BatchNormalization()(x)
x = Conv3D(256, 3, activation="relu")(x);     x = MaxPool3D(2)(x); x = BatchNormalization()(x)
x = GlobalAveragePooling3D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation="sigmoid")(x)
```

Referencias de ese ejemplo: ~1.35 M parámetros, `ExponentialDecay(1e-4, decay_steps=100000, decay_rate=0.96, staircase=True)`, `binary_crossentropy`, EarlyStopping con patience 15.

Las tres versiones que pide P2:

1. **Base:** 2 × (Conv3D + MaxPool3D) → `Flatten` → `Dense` → sigmoid. Con Flatten sobre 128×128×64 los parámetros de la capa densa explotan; anótalo como observación.
2. **Final (paper):** el bloque de arriba.
3. **Solo convolucional con GAP:** reemplazar `Flatten`+`Dense` por `GlobalAveragePooling3D` directo a sigmoid (o con una conv 1×1×1 final). Compara parámetros y `val_acc`. Recuerda que "no es lo mismo una capa con 1024 filtros que con 128": documenta el número de parámetros de `model.summary()`.

## 5. Repaso teórico por unidad (fórmulas a dominar)

**S1–S2 · Regresión y gradiente descendente**
- Pérdida MSE: L = (1/2m) Σ (f(xᵢ) − yᵢ)²; el 1/2 simplifica la derivada.
- Ecuación normal: β = (XᵀX)⁻¹Xᵀy; es costosa para muchos parámetros, por eso se usa GD.
- Actualización: θ ← θ − lr·∇L. lr grande diverge/salta, lr pequeño converge lento. Normalizar features antes.
- Una neurona lineal = regresión; con sigmoid/softmax = clasificación.

**S3 · Clasificación y redes**
- Softmax: e^{zᵢ}/Σe^{zⱼ}; cross-entropy contra one-hot. En PyTorch `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss` (no aplicar softmax antes).
- Backprop: regla de la cadena sobre el grafo computacional; forward guarda activaciones.
- Bucle: mini-batch → forward → loss → backward → update.
- Activaciones: sigmoid/tanh saturan; ReLU no satura en x>0 pero tiene neuronas muertas; Leaky/PReLU/ELU las mitigan. Sin no linealidad las capas colapsan en una lineal.

**S4 · Inicialización, regularización, optimización**
- W=0 → simetría; W pequeño → activaciones a 0; W grande → saturación.
- Xavier (tanh/sigmoid) vs He (ReLU, varianza 2/n_in).
- L2: penaliza ‖W‖ (no el bias). BN: normaliza por capa, capa previa sin bias. Dropout: apaga neuronas al azar en entrenamiento (efecto ensemble); en inferencia no se aplica.
- Momentum (v = μv + lr·∇), RMSProp (escala por cache de gradientes²), Adam (m y v con bias-correction). LR decay: step, exponencial, por tiempo.

**S5 · CNN**
- Tamaño de salida: (N − F + 2P)/S + 1. Ejemplo: 32, F=5, P=0, S=1 → 28.
- Parámetros de una conv: (k·k·C_in + 1)·C_out. Parámetros compartidos e invarianza a traslación.
- Pooling reduce resolución sin parámetros.
- Línea de tiempo: LeNet-5 (1998) → AlexNet (2012: ReLU, dropout) → VGG (3×3 apilados) → GoogLeNet (Inception 1×1) → ResNet (skip aditivas).

**S6 · Transfer learning y destilación**
- Congelar `features` y reentrenar el clasificador (pocos datos, dominio similar) vs fine-tuning (más datos o dominio distinto, lr bajo).
- Preprocesar como el modelo origen (224×224, media/std de ImageNet).
- Destilación: L = (1−α)·CE + α·T²·KL(softmax(z_s/T) ‖ softmax(z_t/T)).

**S7 (posiblemente fuera del parcial)** · GAP y CAM, ResNet (suma) vs DenseNet (concatenación), U-Net, detección (R-CNN → Fast → Faster → Mask, YOLO), NMS con IoU.

## 6. Métricas y metodología (Unidad 5)

- Accuracy = (TP+TN)/N; precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R).
- Curvas de train/val: brecha grande = sobreajuste; ambas altas = subajuste.
- Split estratificado para mantener proporción de clases; nunca ajustar sobre el conjunto de test.
- Nota de rigor: el examen 2026-1 usa el mismo conjunto como `validation_data` y para reportar resultados finales; si te preguntan, menciona que sería mejor un split train/val/test.

## 7. Pendientes que conviene cerrar antes del examen

- [ ] `S2/1INF52_Lab01_Parte2.ipynb`: 1-NN vectorizado (NumPy/TF/PyTorch), sin resolver.
- [ ] `S4/1INF52_Lab02.ipynb`: regresión manual vs `torch.nn`, sin resolver.
- [ ] Rehacer el examen 2026-1 completo en Colab con GPU, cronometrado (P2 es la mitad de la nota).
- [ ] Practicar de memoria el bloque Conv3D–MaxPool3D–BN y la versión GAP.
- [ ] Confirmar fecha, formato y política de IA con los profesores.
- [ ] Las semanas 8 y 10–15 no tienen contenido; el parcial no las cubriría, pero verifica el alcance de S7.

## 8. Plan sugerido

1. Día 1: repasar sección 5 (S1–S4) y hacer los dos laboratorios pendientes.
2. Día 2: S5–S6, calcular a mano tamaños y parámetros de una CNN; redactar respuestas de P1, P3, P4, P5 sin mirar.
3. Día 3: rehacer el examen 2026-1 completo, cronometrado, con las tres versiones del modelo.
4. Día 4: repaso de errores, fórmulas y confirmación de logística.
