# Guion de exposición — Examen Parcial INF659 (2026-2)

**Cómo usarlo:** lo que va en cita (`>`) se **lee tal cual**. Las líneas en negrita con ▶ son **acciones en pantalla** y no se leen.
**Pantalla:** compartir pantalla completa. PowerPoint en modo presentación (F5) y Colab en una sola pestaña del navegador, con el notebook ya ejecutado. Se cambia con Alt+Tab.
**Duración:** ≈8:45 hablado + 3 visitas a Colab de ≈10 s = ≈9:15 (leyendo a 130 palabras por minuto).

| # | Diapositiva | Tiempo | Pantalla |
|---|---|---|---|
| 1 | Portada y contexto | 0:45 | PowerPoint |
| 2 | P1 · Modelo base | 0:50 | PowerPoint → **Colab** (`model.summary()`) → PowerPoint |
| 3 | P1 · Predicción vs realidad | 0:45 | PowerPoint (el gráfico de curvas ya está en la diapositiva) |
| 4 | P2 · Ablación | 0:40 | PowerPoint |
| 5 | P2 · Por qué cambian | 1:00 | PowerPoint |
| 6 | P3 · Slice-wise | 0:45 | PowerPoint |
| 7 | P3 · FE vs FT | 0:55 | PowerPoint → **Colab** (`resultados`) → PowerPoint |
| 8 | P4 · Grad-CAM | 1:05 | PowerPoint → **Colab** (figura) → PowerPoint |
| 9 | P5 · Fiabilidad y BatchNorm | 0:45 | PowerPoint |
| 10 | P5 · Métrica clínica | 0:50 | PowerPoint |
| 11 | Cierre | 0:30 | PowerPoint |

---

## 1 · Portada y contexto

**▶ MOSTRAR: diapositiva 1 ("Clasificación de TC 3D: COVID vs normal").**

> "Buenas noches, soy Jorge Chamorro. En este video presento mi solución del examen parcial de INF659: clasificar tomografías 3D como COVID o normales. Usé 200 volúmenes, 100 de cada clase, de 128 por 128 por 64. La partición estratificada es de 170 volúmenes para entrenar y 30 para prueba, y ese conjunto de prueba hizo también de validación, así que las métricas son optimistas. Todos los resultados son de 10 épocas, por la caída de la sesión de Colab y por tiempo. Antes hice una corrida de 30 épocas, que declaro en el notebook."

---

## 2 · P1 · Modelo base

**▶ MOSTRAR: diapositiva 2 ("P1 · Modelo base: 98 % de parámetros en las capas finales").**

> "El modelo base tiene tres bloques de convolución 3D, max pooling y BatchNorm, con 16, 32 y 32 filtros. Después hay una capa de aplanado, una capa densa de 64 neuronas, Dropout de 0,3 y una salida sigmoide."

**▶ CAMBIAR A COLAB (Alt+Tab): celda de P1 cuya última línea es `model.summary()`. MOSTRAR la tabla de salida `Model: "functional"` y desplazarla hasta la última línea: `Total params: 2,450,881`.**

> "Este es el resumen del modelo: el total es 2 450 881 parámetros. Cada convolución tiene filtro al cubo por canales de entrada, más 1, por canales de salida: la primera da 448, más 64 de la BatchNorm, 512. El 98 por ciento está en las capas finales: el aplanado entrega 14 por 14 por 6 por 32, es decir 37 632 valores, a la capa densa de 64 neuronas."

**▶ VOLVER a la diapositiva 2 y avanzar a la 3.**

---

## 3 · P1 · Predicción vs realidad

**▶ MOSTRAR: diapositiva 3 ("P1 · Predicción vs realidad: sobreajuste sí, mínimo no"). Señalar con el cursor el gráfico de curvas a la izquierda (loss a la izquierda, accuracy a la derecha).**

> "Predije sobreajuste, con una brecha de al menos 10 puntos, y se cumplió: en la época 10, exactitud de 0,841 en entrenamiento contra 0,700 en validación. La pérdida de validación no mostró un mínimo seguido de subida: el menor valor es 0,500 en la época 9 y sube a 0,528 en la 10. Lo que no previ: en las épocas 1 a 4 la exactitud de validación se queda en 0,50 con pérdida entre 1,02 y 1,21. Eso no es sobreajuste sino una diferencia entre entrenamiento e inferencia; mi hipótesis es la BatchNorm con lote de 2, que no comprobé."

---

## 4 · P2 · Ablación

**▶ MOSTRAR: diapositiva 4 ("P2 · Ablación: base densa vs residual vs GAP"). Señalar las filas de la tabla al nombrarlas.**

> "Comparé tres variantes con los mismos filtros, tasa de aprendizaje, épocas y criterios de parada. Mis hipótesis: la base sobreajusta; la residual mejora poco; y GAP, que promedia cada mapa, sobreajusta menos pero puede quedarse corta. Resultados: la base llega a 0,700 de exactitud, con F1 de 0,743, en la época 7; la residual se queda en 0,500 con F1 de 0, no aprendió; y GAP llega a 0,667 con F1 de 0,737 en la época 8, con 42 337 parámetros frente a 2,45 millones."

---

## 5 · P2 · Por qué cambian los resultados

**▶ MOSTRAR: diapositiva 5 ("P2 · Por qué cambian los resultados"). Señalar el gráfico de la izquierda (residual) al empezar y el de la derecha (GAP) en el segundo párrafo.**

> "En la residual, la pérdida baja de 2,07 a 0,693, que es lo que se obtiene al predecir siempre 0,5: clasifica todo como normal. Pasó también en la corrida de 30 épocas. Mi hipótesis, que no verifiqué, es que el bloque termina sin normalización y la capa densa recibe 50 400 valores sin normalizar. Eso no cuestiona a las residuales en general: ayudan en redes profundas y esta es poco profunda."

**▶ SEÑALAR el gráfico de la derecha (GAP).**

> "En GAP, promediar cada mapa de activación reduce los parámetros unas 58 veces. No sobreajusta, pero aprende lento: la exactitud de validación se queda en 0,50 hasta la época 7. Elegí la base como mejor de las tres desde cero, pero la diferencia con GAP es un solo error, 21 contra 20 aciertos: es ruido."

---

## 6 · P3 · Estrategia slice-wise

**▶ MOSTRAR: diapositiva 6 ("P3 · Transferencia de aprendizaje por cortes con ResNet50"). Recorrer con el cursor las seis cajas de izquierda a derecha.**

> "Para la transferencia de aprendizaje usé ResNet50, preentrenada en ImageNet, aplicada por cortes: 16 cortes por volumen, uno de cada cuatro. Cada corte en gris se replica a tres canales y se preprocesa como pide ResNet50. De cada corte salen 2048 características, que promedio antes de la salida sigmoide. Lo hice por cortes y no extendiendo los filtros a 3D porque así reutilizo los pesos 2D sin reescribirlos. La BatchNorm de la red preentrenada la dejé congelada: con lote de 2 y 16 cortes de solo 2 pacientes, sus estadísticas no serían representativas."

---

## 7 · P3 · Feature extraction vs fine-tuning

**▶ MOSTRAR: diapositiva 7 ("P3 · Fine-tuning supera a feature extraction y al modelo base"). Señalar la tabla.**

> "Con feature extraction, es decir, entrenando solo la capa final, con 2 049 parámetros y tasa de aprendizaje de 0,001, la exactitud sube a 0,867, 26 aciertos de 30. Con fine-tuning, es decir, ajustando también el último bloque de la red, con tasa de 0,0001, llega a 0,967, 29 de 30."

**▶ CAMBIAR A COLAB (Alt+Tab): celda de P3 cuya última línea es `print(f"{k:24s} params=...` (dentro de `for k, v in resultados.items():`). MOSTRAR la salida impresa: las cinco líneas `a_base`, `b_residual`, `c_gap`, `tl_feature_extraction` y `tl_finetuning`. Señalar la última (`tl_finetuning ... acc=0.967 f1=0.966 mejor_ep=10`).**

> "Esta es la salida del notebook: fine-tuning, 0,967 con F1 de 0,966, en la época 10."

**▶ VOLVER a la diapositiva 7.**

> "ImageNet son fotos a color y esto son cortes de TC en gris: hay una brecha de dominio; las características simples se transfieren y el fine-tuning adapta las más complejas. Reservas: una sola corrida, la mejor época es la última, y la elegí con el mismo conjunto de prueba, así que 0,967 es optimista."

---

## 8 · P4 · Grad-CAM

**▶ MOSTRAR: diapositiva 8 ("P4 · Grad-CAM: el calor se reparte sin foco clínico").**

> "Apliqué Grad-CAM a la última capa convolucional del modelo base. No usé CAM porque la base no tiene promedio global; Grad-CAM lo generaliza usando el promedio de los gradientes como peso de cada mapa de activación."

**▶ CAMBIAR A COLAB (Alt+Tab): celda de P4 cuya última línea es `plt.tight_layout(); plt.show()`. MOSTRAR primero las 4 líneas impresas (`Acierto idx=5 …`, `Acierto idx=13 …`, `Error idx=11 …`, `Error idx=2 …`) y luego desplazarse a la figura de 4 filas por 3 columnas. Señalar cada fila al nombrarla.**

> "Cuatro casos. Primera fila, acierto anormal, caso 5, probabilidad 0,99: solo 11 por ciento del calor en el borde. Segunda fila, acierto normal, caso 13: 52 por ciento, difuso en pared, fondo y mesa. Tercera y cuarta fila, dos falsos positivos: el caso 11, probabilidad 0,88, con 46 por ciento, mira pared, hombro y esternón; el caso 2, probabilidad 0,77, con 52 por ciento, mira pared y fuera del cuerpo."

**▶ VOLVER a la diapositiva 8.**

> "El borde que medí ocupa cerca del 51 por ciento del área, así que 52 por ciento de calor ahí no prueba sesgo: indica calor sin foco. Limitación: no lo hice sobre el mejor modelo, el de fine-tuning."

---

## 9 · P5 · Fiabilidad y BatchNorm

**▶ MOSTRAR: diapositiva 9 ("P5 · Con 30 casos, las diferencias son poco fiables"). Señalar la tabla de intervalos a la derecha.**

> "Con 30 casos, un error mueve la exactitud 3,3 puntos. Con intervalos de confianza de Wilson al 95 por ciento, la base va de 0,521 a 0,833 y GAP de 0,488 a 0,808: no se pueden distinguir. Fine-tuning va de 0,833 a 0,994 y apenas se separa de la base, porque su límite inferior coincide con el límite superior de la base."

> "Sobre la BatchNorm con lote de 2: en entrenamiento usa estadísticas de solo dos volúmenes, muy ruidosas; en inferencia usa medias móviles, que se actualizan lento. Es una hipótesis. Alternativas: normalización por grupos o por capa, o acumular gradientes."

---

## 10 · P5 · Métrica clínica y modo de fallo

**▶ MOSTRAR: diapositiva 10 ("P5 · Métrica clínica: sensibilidad y especificidad"). Señalar la fila de cada modelo al nombrarlo.**

> "En tamizaje un falso negativo cuesta más que un falso positivo, porque deja pasar a un paciente. Miro sensibilidad, especificidad y AUC. La base tiene sensibilidad de 0,87 y especificidad de 0,53, con 2 falsos negativos y 7 falsos positivos. GAP sube la sensibilidad a 0,93, pero la especificidad cae a 0,40, con 9 falsos positivos. Fine-tuning tiene sensibilidad de 0,93, especificidad de 1,00 y AUC de 0,97: 1 falso negativo y ninguno positivo. El fallo que vi en Grad-CAM son falsos positivos con alta confianza apoyados en pared y fondo; propondría segmentar el pulmón con una máscara y verificar de nuevo con Grad-CAM."

---

## 11 · Cierre

**▶ MOSTRAR: diapositiva 11 ("Cierre: qué no funcionó y uso de IA").**

> "No funcionó la residual, que colapsó a 0,5 sin que yo verificara la causa; 10 épocas fueron pocas para GAP; y no apliqué Grad-CAM al mejor modelo. Usé Claude para generar código, depurar y redactar borradores; ejecuté todo y decidí yo. Fine-tuning fue lo único que superó claramente a los modelos entrenados desde cero, con la reserva de que son 30 casos y una sola corrida. Gracias."

**▶ DETENER la grabación.**

---

## Glosario de términos del guion

| Se dice | Se evita decir |
|---|---|
| exactitud | accuracy, `val_acc` |
| pérdida (de entrenamiento / de validación) | loss, `val_loss` |
| sensibilidad | recall |
| conjunto de prueba, partición | test, split |
| lote (de 2) | batch |
| tasa de aprendizaje | learning rate, lr |
| aplanado, capa densa | Flatten, Dense |
| brecha de dominio | domain gap |
| transferencia de aprendizaje, por cortes | transfer learning, slice-wise |
| caso 5, caso 13 | `idx` |
| último bloque de la red | `conv5_` |
| última capa convolucional | `conv3d_2` |
| **Se mantienen en inglés** | BatchNorm, max pooling, Dropout, GAP, Grad-CAM, CAM, ResNet50, ImageNet, feature extraction, fine-tuning, F1, AUC, notebook, Colab |

Los nombres exactos del código (`model.summary()`, `tl_finetuning`, etc.) aparecen solo en las líneas ▶, que no se leen.

## Matriz de coherencia (diapositiva ↔ código ↔ figura ↔ salida)

| # | Cifra o afirmación | Salida impresa en Colab | Figura / archivo |
|---|---|---|---|
| 2 | 2 450 881 parámetros; 98 % en la cabeza | `Total params: 2,450,881` (celda de `model.summary()`) | tabla de la diapositiva |
| 3 | train 0,841 / val 0,700; ép. 1–4 val_acc 0,50 | Época 10: `acc 0.8412 val_acc 0.7000`; ép. 1–4 `val_loss` 1,02–1,21 | `P1/P2-1.png` (curvas del base) |
| 4 | a 0,700 · b 0,500 · c 0,667 | celda `resultados` de P3 | tabla |
| 5 | b: loss 0,693; c: 0,50 hasta ép. 7 | `b_residual … epocas 9, mejor_epoca 1` | `P2/P2-1.png`, `P2/P2-2.png` |
| 7 | FE 0,867; FT 0,967; 29/30 | `tl_feature_extraction`, `tl_finetuning`, `[[15 0],[1 14]]` | `P3/P3-1.png`, `P3/P3-2.png` |
| 8 | 11 / 52 / 46 / 52 % | 4 líneas impresas de P4 | `P4/P4.png` |
| 9 | intervalos de Wilson | celda 1 de P5 | gráfico y tabla nativos |
| 10 | recall / especificidad / AUC | celda 2 de P5 | tabla |

## Preguntas que debes poder responder sin leer

- ¿Por qué a la época 10 el lr está en ≈35 %? → 0,9^10 ≈ 0,35 (decay_steps = 85 = una época).
- ¿Por qué el Flatten tiene 37 632 valores? → 14×14×6×32.
- ¿Por qué ln 2 significa que no aprendió? → es la loss de predecir 0,5 siempre.
- ¿Por qué congelaste BatchNorm en el fine-tuning? → batch pequeño, estadísticas de ImageNet más estables.
- ¿Qué pasó con la corrida de 30 épocas? → Fue la primera corrida; se declara en el notebook (predicción de P1 y colapso de la residual). Los resultados mostrados son todos de 10 épocas. No cites "0,867 con 30 épocas" que menciona el notebook en P5: ya no hay evidencia de esa corrida en la carpeta.
- Si preguntan por la val_loss de GAP en la época 10 → el notebook dice que "aún bajaba", pero la curva muestra 0,585 (época 9) y 0,61 (época 10): quedó estable, con una ligera subida. Dilo así.
- ¿Por qué no es válido concluir que fine-tuning "es mejor" con certeza? → n = 30, un error = 3,3 puntos, test usado como validación, una sola corrida.
