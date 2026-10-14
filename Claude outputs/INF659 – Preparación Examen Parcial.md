# INF659 – Preparación para el Examen Parcial (versión actualizada)

Actualizado el 13 de octubre de 2026. Versión original del 29 de septiembre. Esta versión incorpora: el examen parcial real 2026-2 (ya rendido) con sus resultados, las transcripciones de las clases S3 a S7 y S9, el notebook de autoencoder, la carpeta del proyecto (datasets de deforestación) y las lecciones del proceso.

**Aviso de fuentes.** Las transcripciones (`DM S3.txt` … `DM S9.txt`) son de audio automático y tienen errores (por ejemplo "Savile" = Xavier, "Adana-B" = Adam, "Lenet" por ResNet). No traen la fecha de la clase: los archivos figuran actualizados el 13/10/2026, pero eso es la fecha del archivo, no necesariamente la de la clase. Lo dudoso está marcado como **por confirmar**.

---

## 1. Qué fue el examen parcial (real, 2026-2)

- **Modalidad:** 3 horas, en casa, individual, notebook en Colab (TensorFlow/Keras). Se rindió el 6 de octubre de 2026, de 7 a 10 pm.
- **IA:** permitida y esperada. El profesor dijo en clase que no la puede prohibir y que por eso hay video: la evaluación es de comprensión, no de ejecución.
- **Entrega:** notebook ejecutado + enlace al video de sustentación (**máximo 10 min, obligatorio**). El video se sube después de las 3 horas.
- **Regla central:** los puntos de cada parte **se acreditan solo si puedes explicar ese trabajo en el video**. Casi todas las preguntas dependen de los resultados de tu propia corrida (curvas, tabla de ablación, mapas de calor).
- **Otros requisitos:** cuaderno de decisiones (el *por qué* de cada decisión), declaración de uso de IA, reportar lo que NO funcionó, no tocar las semillas, declarar si se reducen épocas o resolución.
- **Peso:** EP = 30 % de la nota final (NF = 0.2·TA + 0.3·EP + 0.5·PC). Total del examen: 20 puntos.

### Estructura y puntajes

| Parte | Pts | Qué pedía |
|---|---|---|
| P1 · Línea base e instrumentación | 3 | Modelo base (conv + pooling + cabeza densa → sigmoide); número de parámetros total y por bloque; **predecir antes de entrenar** cómo se vería la brecha train/val y compararla con la realidad; decisiones de diseño. |
| P2 · Intervención arquitectónica y ablación | 6 | Variantes (a) base, (b) bloques residuales 3D, (c) solo convolucional con GAP; hipótesis previa de cada una; por qué cambian los resultados; cuál elegirías. |
| P3 · Transfer learning 2D→3D | 4 | Estrategia (se eligió slice-wise con ResNet50 de ImageNet); feature extraction vs fine-tuning con números; domain gap; BatchNorm con `batch_size=2`. |
| P4 · Interpretabilidad | 3 | Grad-CAM 3D: ¿el modelo mira los pulmones? Casos: aciertos y errores con mayor confianza. |
| P5 · Diagnóstico y rigor experimental | 4 | Fiabilidad de las diferencias (IC o k-fold); BatchNorm con batch 2 en entrenamiento e inferencia; métrica y costo clínico; modo de fallo y su corrección. |

Datos del problema: MosMedData (100 normal + 100 COVID), volúmenes 128×128×64, HU recortada a [-1000, 400] y normalizada a [0, 1], split estratificado 170/30 (test también usado como validación), `batch_size=2`, Adam con `ExponentialDecay(1e-4, decay_steps=85, decay_rate=0.90)`, `ModelCheckpoint` y `EarlyStopping(val_acc, patience=8)`.

### Resultados de mi corrida (10 épocas, GPU T4)

| Modelo | Parámetros | Exactitud val. | F1 | Mejor época | Nota |
|---|---|---|---|---|---|
| (a) Base densa | 2 450 881 | 0,700 | 0,743 | 7 | train 0,841 → brecha de 14 pts |
| (b) Residual 3D | 3 324 193 | 0,500 | 0,000 | 1 | colapso: pérdida = 0,693 (salida constante 0,5) |
| (c) GAP | 42 337 | 0,667 | 0,737 | 8 | aprende lento (0,50 hasta la época 7) |
| Feature extraction (ResNet50) | 23 589 761 (2 049 entrenables) | 0,867 | 0,867 | 8 | lr 1e-3 |
| Fine-tuning (bloque conv5) | 23 589 761 (14 955 521 entrenables) | 0,967 | 0,966 | 10 | lr 1e-4; 1 falso negativo, 0 falsos positivos |

- **Grad-CAM (modelo base):** 4 casos con 11 % / 52 % / 46 % / 52 % del calor en el borde. El borde (15 % exterior) ocupa ≈51 % del área, así que 52 % no prueba sesgo hacia los bordes: indica calor sin foco.
- **Fiabilidad (Wilson 95 %, n = 30):** base [0,521–0,833], GAP [0,488–0,808], feature extraction [0,703–0,947], fine-tuning [0,833–0,994]. Un error mueve la exactitud 3,3 puntos.
- **Métrica clínica:** base recall 0,87 / especificidad 0,53 / AUC 0,85; GAP 0,93 / 0,40 / 0,71; feature extraction 0,87 / 0,87 / 0,86; fine-tuning 0,93 / 1,00 / 0,97.

### Lecciones del proceso (qué funcionó y qué no)

- **Colab:** usar GPU T4 con RAM alta. La TPU no sirve con este código. Con TPU la sesión se cayó. El cuello de botella son los 30 s por época del aumento de datos en CPU.
- **Reducir a 10 épocas** (de 30) por la caída de sesión y por tiempo; **declararlo** y aplicarlo igual a todas las variantes. Con 10 épocas `EarlyStopping` (patience 8) casi no puede actuar.
- **Errores técnicos:** faltaban imports (`Dense` y otros); `tf.numpy_function` pierde la forma estática, se corrige con `scan.set_shape((width, height, depth))`; cuidado con checkpoints antiguos (cargar el más reciente con `max(glob(...), key=os.path.getmtime)`).
- **Residual (b) colapsó dos veces** (30 y 10 épocas). Hipótesis, **no verificada**: el bloque termina sin BatchNorm y la capa densa recibe 50 400 valores sin normalizar. Se reportó como intento fallido, sin ocultarlo.
- **BatchNorm con batch 2:** explica la hipótesis de `val_acc` = 0,50 y pérdida > 1 en las primeras épocas (estadísticas ruidosas en entrenamiento y medias móviles lentas, momentum 0,99, en inferencia). No se aisló experimentalmente.
- **Predicción previa:** olvidé que el lr decae (≈35 % a la época 10) y sobreinterpreté el 52 % de calor en el borde; ambas cosas se corrigieron. Lección: calcular la base (área del borde, decaimiento del lr) antes de interpretar.
- **Honestidad:** el test hizo de validación, así que las métricas son optimistas; con 30 volúmenes cada error pesa 3,3 puntos; la mejor época del fine-tuning es la última.
- **Entrega:** el video y la presentación (11 diapositivas, ≈9 min) se prepararon con guion literal, coherente con el notebook y con las cifras impresas. Evitar términos de código en voz alta; usar inglés solo para BatchNorm, GAP, Grad-CAM, feature extraction, fine-tuning, F1 y AUC.

---

## 2. Examen de práctica 2026-1 (referencia)

Caso muy parecido: CT scans 3D de pulmón, 100 volúmenes por clase, 128×128×64, split 85/15, `batch_size=2`, Adam con `ExponentialDecay`.

| Pregunta | Pts | Qué había que hacer |
|---|---|---|
| P1 | 2 | Analizar el data augmentation (rotación aleatoria ±5/10/20°). |
| P2 | 10 | Modelo base (2 pts); versión final según el paper (5 pts); versión solo convolucional con GAP (3 pts). |
| P3 | 2 | Elegir parámetros de `ExponentialDecay` y explicar el efecto del lr. |
| P4 | 2 | Elegir `epochs` y `patience` y explicar su efecto. |
| P4bis | 2 | Conv 3D vs 2D; si el transfer learning ayudaría y qué características debería tener el modelo origen. |
| P5 | 2 | Por qué exactitud es adecuada aquí y diferencia con F1. |

El examen real 2026-2 **cambió el enfoque**: del "arma el modelo" al "diseca, interviene, diagnostica y defiende" con video obligatorio.

### Respuestas modelo (práctica)

**Data augmentation.** La rotación aleatoria solo transforma los volúmenes existentes; no agrega volúmenes al corpus, sino variantes distintas en cada época. Reduce el sobreajuste con ~170 volúmenes de entrenamiento. Perjudica si el ángulo es excesivo (anatomía irreal) y suma costo de CPU. Se aplica solo a train.

**ExponentialDecay.** `lr = lr0 · decay_rate^(step/decay_steps)`. Con `decay_steps` = 85 (pasos de una época con batch 2) y decay 0,90, a la época 10 el lr queda en 0,9^10 ≈ 35 %. Si `decay_steps` es muy grande frente a los pasos reales, el lr casi no decae.

**epochs y patience.** `epochs` es el tope de pasadas; `patience` son las épocas sin mejora de `val_acc` toleradas antes de detener. Muy bajo corta antes de salir de mesetas; muy alto gasta cómputo.

**Conv 3D vs 2D.** El filtro es k×k×k×C_in y se desliza en ancho, alto y profundidad; captura contexto entre cortes, con más parámetros y memoria. Un modelo ImageNet 2D requiere reformular la entrada y sufre desajuste de dominio.

**Exactitud vs F1.** Con clases balanceadas y split estratificado, la exactitud es razonable. F1 (media armónica de precisión y recall) es clave con clases desbalanceadas o costos distintos de error (en diagnóstico, el recall de la clase anormal es crítico).

---

## 3. Arquitectura de referencia (P2 de 2026-1)

El examen apunta al paper arXiv:2007.13224 (Zunair et al.) y al repositorio `hasibzunair/uniformizing-3D`. El diseño siguiente es el del ejemplo oficial de Keras "3D image classification from CT scans" (verifícalo contra el paper):

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

Referencias: ~1,35 M parámetros, `ExponentialDecay(1e-4, decay_steps=100000, decay_rate=0.96, staircase=True)`, `binary_crossentropy`, EarlyStopping con patience 15.

**Modelo base del examen 2026-2** (tres bloques Conv3D 3 + MaxPool3D(2) + BatchNorm con 16/32/32 filtros, sin padding): 128×128×64 → 63×63×31 → 30×30×14 → 14×14×6 antes del Flatten, es decir 14·14·6·32 = 37 632 valores a `Dense(64)`.

---

## 4. Repaso teórico por unidad (con lo aprendido en clase)

### S1–S2 · Regresión y gradiente descendente
- Pérdida MSE: L = (1/2m) Σ (f(xᵢ) − yᵢ)²; el 1/2 simplifica la derivada. Ecuación normal: β = (XᵀX)⁻¹Xᵀy.
- Actualización: θ ← θ − lr·∇L. lr grande diverge; lr pequeño converge lento. Normalizar features antes.

### S3 · Clasificación, regresión logística, softmax y redes (detalle de clase)
Hilo del profesor: regresión lineal → clasificación binaria → multiclase → capas → activaciones → backpropagation → minilotes. Cada paso se construye sobre el anterior.

- **Repaso previo:** regresión = predecir un valor numérico; clasificación = predecir una clase (valor nominal). Se corrigió un error de fórmula en la pérdida (faltaba el 1/m). En el código, el gradiente y la predicción se reescriben como multiplicación de matrices: menos código y operación ideal para paralelizar en GPU. Ejemplos: casas (simple y múltiple), diabetes (442 registros, 296 para entrenamiento).
- **Por qué no usar una recta para clasificar:** con corte en f(x) > 0,5 funciona en un caso simple, pero (1) al agregar datos nuevos (tumores más grandes) la recta se mueve y el corte cambia (0,5 → 0,3): se querría un corte constante; (2) con clases desbalanceadas empeora; (3) la salida puede ser negativa o mayor que 1, sin interpretación. Solución: acotar la salida entre 0 y 1.
- **Sigmoide:** g(z) = 1/(1+e^(−z)). Si z → ∞, g → 1; si z → −∞, g → 0 (nunca los alcanza). Se compone con la función lineal: z = wx + b, h(x) = g(wx + b). Con salida acotada el corte es siempre 0,5: h(x) > 0,5 → clase 1; si no, clase 0. Es un clasificador **binario** y se llama **regresión logística** ("suena contradictorio": clasifica, pero se construye sobre una regresión).
- **Función de pérdida de la logística:** el MSE sobre la sigmoide da una función **no convexa** (varios mínimos locales): el descenso por gradiente puede quedar atrapado según el punto de inicio. Convexa = un solo mínimo, que es el global. Por eso se usa otra pérdida (entropía cruzada binaria): −log(h(x)) si y = 1 y −log(1 − h(x)) si y = 0; se combinan en una sola expresión con y y (1 − y). Si el modelo acierta, el costo tiende a 0; si se aleja de la etiqueta, el costo crece hacia el infinito (h = 0 con y = 1). Costo total = promedio sobre el conjunto de entrenamiento.
- **Gradientes:** las derivadas de la pérdida logística tienen la misma **forma** que las de la regresión lineal; lo que cambia es h(x): lineal h = wx, logística h = 1/(1+e^(−(wx+b))). Por eso el mismo algoritmo de descenso por gradiente sirve cambiando solo la hipótesis.
- **Multiclase (Iris: 3 clases, 4 características):** etiquetas **one-hot** (setosa = 1,0,0; versicolor = 0,1,0; virginica = 0,0,1) en vez de 0/1/2. Una función lineal por clase: entradas 4×n, salidas 3×n, matriz de pesos 3×4 (una fila por clase) + sesgos. Se puede absorber el sesgo con una columna extra y x₀ = 1. Es una **capa densa / totalmente conectada** (cada entrada conecta con cada salida). Cuatro representaciones equivalentes: función, matriz, grafo y regresiones multivariadas separadas. Cada clase necesita pesos distintos (hiperplanos que separan regiones). Con 20–30 clases hay tantas funciones lineales como clases, calculadas en paralelo (es una sola multiplicación de matrices).
- **Logits y softmax:** las salidas de las funciones lineales se llaman **logits** (pueden ser muy negativos o positivos, sin interpretación directa). Softmax: exponencial de cada logit dividida entre la suma de todas → valores entre 0 y 1 que suman 1 (interpretables como probabilidades); se elige la clase con mayor valor. Es la generalización multinomial de la sigmoide. Ejemplo de clase: salida 0,99 para la clase correcta y error de entropía cruzada ≈ 0,007.
- **Entropía cruzada (categórica):** −Σ yᵢ·log(ŷᵢ), con y one-hot. Su nombre viene de la forma parecida a la entropía.
- **Backpropagation (Rumelhart, Hinton y Williams, 1986):** el artículo que puso fin al "invierno de la IA". Ciclo de entrenamiento de toda red: **forward propagation** (calcula la salida y la pérdida) → **backpropagation** (calcula los gradientes hacia atrás con la **regla de la cadena**; responde "quién es culpable del error y en qué magnitud") → **actualización de pesos**; se repite. `fit` en Keras ejecuta esos tres pasos. Ejemplo de clase: f = (x + y)·z con x = −2, y = 5, z = −4; ∂f/∂z = x + y = 3, ∂f/∂y = z·1 = −4 (cadena).
- **Minilotes:** no se procesa todo el dataset por iteración (inviable con datasets enormes); en cada iteración se toma un lote (minilote). La pérdida sube y baja entre iteraciones (lotes poco representativos), pero baja en tendencia = el modelo aprende. Tarda algo más pero gasta menos recursos.
- **Por qué activaciones:** componer funciones lineales da otra función lineal (ejemplo: y₁ = 2x + 10 y y₂ = 3y₁ − 5 → y₂ = 6x + 25), así que muchas capas sin no linealidad se reducen a una. La activación no linealiza cada nodo y permite que cada uno se especialice en un patrón ("se activa").
  - **Sigmoide:** se satura con valores altos (gradiente ≈ 0), sus salidas son siempre positivas (no centradas en cero) y la exponencial es costosa.
  - **Tanh:** parecida a la sigmoide pero centrada en cero; también satura.
  - **ReLU:** identidad si x > 0, cero si x < 0. No satura en la parte positiva, es muy barata (sin cálculo) y muy usada en deep learning y en imágenes. Problema: en x < 0 el gradiente es 0 (neuronas muertas). Alternativas: Leaky ReLU (constante pequeña), PReLU (la red aprende el parámetro), ELU.
  - **Softmax:** solo en la **capa de salida** (normaliza sobre las clases); no en capas intermedias.
  - No aplicar BatchNorm ni Dropout en la última capa.
- **Cuántas capas:** no hay número fijo; depende de los datos y del hardware. Pocas capas → modelo simple; demasiadas → modelo complejo y sobreajuste. Ejemplo de clase: ~100 000 datos, unas 6 capas intermedias.
- **Keras (Iris):** `Sequential` con una capa densa de 10 nodos (entrada de 4) y una capa densa de 3 con softmax; `compile` con descenso por gradiente (lr = 0,01), pérdida de entropía cruzada y métrica de exactitud; `fit` con conjunto de entrenamiento, de prueba, épocas y tamaño de lote. TensorFlow es una capa intermedia entre el código y los procesadores (GPU/TPU); hay una versión en PyTorch en el cuaderno para comparar.
- **Examen (aviso del profesor):** en el examen se cuidará que no haya acceso a asistentes de IA abiertos ("vamos a asegurarnos que el cloud no esté abierto").
- **Próxima sesión (S4):** cómo optimizar el entrenamiento (lo que ya figura abajo).

### S4 · Inicialización, regularización y optimización (detalle de clase)
- **Inicialización:** todo en cero → todas las neuronas hacen lo mismo (mismas gradientes). Pesos pequeños → activaciones y gradientes tienden a 0 con la profundidad (peor con sigmoide); pesos grandes con tanh → saturación en ±1 y gradiente 0. **Xavier/Glorot** (2010) mantiene la varianza de las activaciones entre capas (para tanh/sigmoide); **He** (2015) para ReLU, porque solo cuenta la mitad positiva. En Keras: `glorot_normal` y `he_normal`. Sospechar de la inicialización cuando la pérdida demora en caer.
- **Sobreajuste / subajuste:** sobreajuste = pérdida de train baja y de validación alta (gran brecha); subajuste = ambas altas (modelo simple). Remedios del subajuste: más complejidad, más datos, entrenar más.
- **Regularización L2/L1:** se añade λ·R(W) a la pérdida; L2 = suma de cuadrados, L1 = valores absolutos, elastic net = ambas. Distribuye la carga entre pesos (ejemplo del profesor: w=(1,0,0,0) y w=(.25,.25,.25,.25) dan la misma salida; L2 prefiere el segundo). λ ≈ 0 no regulariza; λ muy alto subajusta. Se regulariza W, no el bias (el bias solo desplaza la función; podría regularizarse, pero no tiene sentido). Valores razonables en clase: 0,001; λ = 0,1 penaliza demasiado.
- **BatchNorm:** normaliza activaciones con media y desviación del batch y luego aplica γ y β aprendibles (la red puede deshacer la normalización si le conviene). Con BatchNorm usar `use_bias=False` en la capa previa. No exige batches grandes, pero con batch 2 las estadísticas son ruidosas.
- **Dropout:** apaga nodos al azar en cada iteración (efecto ensamble); un nodo apagado recibe gradiente 0; en inferencia se usan todas las unidades. Cuánto usar y si combinar con BatchNorm es empírico ("alquimia", según el profesor).
- **Optimizadores:** SGD (oscila, se atasca en puntos de silla), Momentum (0,9 típico), Nesterov, AdaGrad (acumula gradientes²; su tasa tiende a 0), RMSProp (memoria con decay; viene de una diapositiva de Hinton) y **Adam** (momentum + RMSProp, con corrección de sesgo porque M y V empiezan en 0; β ≈ 0,9 y 0,999 por defecto).
- **Learning rate:** nunca constante: empezar alto y decrecer (step, exponencial, por tiempo). **LR Finder** (lr crece en cada paso y se mira la pérdida): hacerlo por cada arquitectura. **LR cíclico** (2–4 veces menos épocas según el paper) y **snapshot ensembles** (guardar el modelo en cada mínimo del ciclo; mejora pequeña, 0,9645 → 0,9690 en el ejemplo del alumno).
- **Cálculo de iteraciones:** 60 000 / 256 = 234,375 → 235 iteraciones por época (la última de 96). Un batch sesgado produce gradientes sesgados: la curva de pérdida sale ruidosa.
- **Normalizar la entrada** (÷255): sin eso, las buenas inicializaciones rendían ≈0,10.

### S5 · Convoluciones y arquitecturas
- Tamaño de salida: (N − F + 2P)/S + 1. Ejemplos: 5×5 con kernel 3×3 sin padding → 3×3; 7×7 con kernel 3×3 y padding 1 → 7×7; `same` calcula el padding solo (kernel 5 → padding 2).
- Parámetros de una conv: (k·k·C_in + 1)·C_out. Ejemplos del profesor: 16 filtros 3×3×1 = 160; 32 filtros 3×3×16 = 4 640; densa de 800 a 10 = 8 010 (total del ejemplo 12 810). La profundidad del filtro iguala la de la entrada; filtros 1×1 toman la profundidad del bloque entrante.
- Max pooling 2×2 stride 2 reduce 224×224×64 a 112×112×64 (1/4 de operaciones). Kernel 3, 5 o 7 (3 para imágenes pequeñas); más grande no aporta mucho y cuesta.
- **LeNet:** 32×32×1 → conv 6 filtros 5×5 (28×28×6) → pool (14×14×6) → conv 16 filtros 5×5 (10×10×16) → pool (5×5×16) → flatten → 120 → 84 → 10.
- **AlexNet (2012):** primer filtro 11×11 stride 4 → 55×55; ReLU, dropout, data augmentation, L2, dos GPU; error top-5 ≈15,4 %. **VGG:** todo con filtros 3×3 (VGG16 mejor que VGG19). **GoogLeNet/Inception:** módulo con 1×1, 3×3, 5×5 y pooling en paralelo; 1×1 como cuello de botella; salidas auxiliares contra el gradiente que se desvanece. **ResNet (2015):** conexión residual (sumar X dos capas adelante); BatchNorm, inicialización He, sin dropout; error top-5 de 3,6 % con 152 capas; redes planas profundas empeoraban, las residuales no.
- **Precaución:** el profesor dijo que ReLU es "de 2014"; ReLU es de ≈2010 y se popularizó con AlexNet (2012).

### S6 · Transfer learning y destilación
- **Receta:** congelar la parte convolucional y reemplazar/reentrenar la última capa densa (pocos datos, dominio similar); fine-tuning (más datos o dominio distinto, lr bajo). Preprocesar como el modelo origen (resize a 224, normalización de ImageNet). Para inferir desde un `.pth`: cargar la arquitectura con `weights=None` y luego los pesos.
- **LR diferenciado:** lr pequeño para la parte convolucional y mayor para la densa; un solo lr puede causar **olvido catastrófico**. En el notebook: 1e-4 convolucional y 1e-3 densa.
- **Resultados de clase (CIFAR-10 light):** modelo propio ≈26 % test; VGG16 con TL ≈69 %; ResNet18 con TL ≈64 %. VGG pesa más que ResNet por su cabeza densa.
- **Jerarquía de características:** capas iniciales detectan bordes y colores; las medias, formas; las finales, partes del objeto.
- **Destilación (Hinton):** el profesor grande enseña al alumno pequeño con logits suavizados por temperatura, softmax(z/T) con T > 1, para transmitir la "información oscura" de las clases pequeñas. El tamaño del alumno no cambia. Alternativa: cuantización a int8. Fórmula general de pérdida (de apuntes previos, **no dictada explícitamente en clase**): L = (1−α)·CE + α·T²·KL(softmax(z_s/T) ‖ softmax(z_t/T)).

### S7 · Arquitecturas más allá de la clasificación (entró en el examen: GAP y Grad-CAM)
- **GAP / red totalmente convolucional:** se eliminan las densas; el último bloque tiene tantos mapas como clases y el GAP promedia cada uno. Menos parámetros, y los mapas dan interpretabilidad tipo CAM. Parámetros: AlexNet ≈62 M (más de la mitad en FC), VGG16 138 M (FC1 = 102 M), ResNet50 ≈25 M.
- **MobileNet:** convolución depthwise + pointwise. Ejemplo: entrada 64×64×32 y salida 64×64×8: la conv normal ≈9,4 M de operaciones; depthwise + pointwise ≈2,2 M; se pierde ≈1 punto de exactitud.
- **ResNet** suma (residuos); **DenseNet** concatena todas las capas anteriores.
- **Segmentación:** clasificar cada píxel. **U-Net:** encoder–decoder en U con skip connections por concatenación (copy and crop); filtros 64→1024 en el cuello; el original usa convs sin padding (572→570). Requiere máscaras etiquetadas por un experto.
- **IoU** = |A∩B| / |A∪B| (ground truth contra predicción). **NMS:** entre cajas que se solapan con IoU > umbral (ej. 0,7) se conserva la de mayor probabilidad.
- **Detección:** clasificación + regresión de (x, y, w, h). R-CNN (Selective Search) → Fast R-CNN → Faster R-CNN (RPN entrenable) → Mask R-CNN; YOLO en una sola pasada.

### S9 · Fundamentos estadísticos y autoencoders (prof. Rodrigo Uribe)
- **Probabilidad:** independientes P(A∩B) = P(A)·P(B); excluyentes P(A∩B) = 0; condicional P(A|B) = P(A∩B)/P(B).
- **Información** I = log₂(1/p) (moneda 1 bit; dado ≈2,58 bits; dos dados ≈5,17 bits). **Entropía** = sorpresa promedio (uniforme → alta, concentrada → baja). **Entropía cruzada** = sorpresa de los datos respecto a una creencia; es la pérdida de clasificación (one-hot como creencia).
- **Falsabilidad (Popper):** importan más los casos negativos que los positivos. Frecuentista (parámetros fijos, verosimilitud) vs bayesiano (parámetros con distribución, se actualizan creencias). En ML el estándar de oro es el test set. "All models are wrong, but some are useful." Libro recomendado: *Modeling Mindsets*.
- **Maldición de la dimensionalidad:** el volumen crece exponencialmente y los puntos se vuelven equidistantes; hacen falta exponencialmente más datos.
- **PCA:** código z = Uᵀ(x−μ), reconstrucción x̃ = Uz + μ, U ortonormal; minimizar el error de reconstrucción equivale a maximizar la varianza proyectada; el subespacio óptimo son los autovectores principales de la covarianza.
- **Autoencoder:** red que predice su propia entrada con un cuello de botella. **Un autoencoder lineal con MSE equivale a PCA**; el no lineal aprende variedades y códigos más potentes. El autoencoder simple no da dimensiones interpretables. VAE solo se vio como concepto (espacio latente bayesiano), sin ELBO ni KL.
- **Notebook de ejemplo:** FashionMNIST, MLP 784→128→64→32 (ReLU) y decoder 32→64→128→784 (Sigmoid), MSE, Adam 1e-3, 10 épocas, `batch_size=256`; pérdida 0,0753 → 0,0468. Decodificar vectores aleatorios no garantiza muestras realistas (no hay término de regularización).

---

## 5. Métricas y metodología

- Exactitud = (TP+TN)/N; precisión = TP/(TP+FP); recall (sensibilidad) = TP/(TP+FN); especificidad = TN/(TN+FP); F1 = 2PR/(P+R).
- Curvas train/val: brecha grande = sobreajuste; ambas altas = subajuste.
- **Con n = 30**, un error = 3,3 puntos; usar intervalos (Wilson al 95 %) y no sobreinterpretar diferencias de 1–2 aciertos.
- Split estratificado; nunca ajustar sobre test. Si el test hace de validación, declararlo.
- En tamizaje, el falso negativo cuesta más: optimizar recall con especificidad aceptable y comparar con AUC (independiente del umbral).
- Cuidado con fuga de datos (por ejemplo, frames del mismo video en train y test). "Lo importante del test es que sea insesgado."

---

## 6. Proyecto / Tarea Académica Final y Vibe Quest

- **Proyecto (Tarea Académica Final, doble puntaje):** el profesor pidió tomarlo "de verdad". Permite usar IA (Claude) pero exige entender por dentro: "si estás usando una idea, debes entender por qué". Tres fases: (1) definir un modelo base y buscar un paper; (2) intervenir el modelo; (3) usar el modelo como preentrenamiento para otro problema. Entregables: repositorio en GitHub, documento y aplicación; presentación como póster con defensa. Se pueden usar modelos preentrenados (Meta, Hugging Face) y datasets con segmentación o etiquetas.
- **Fechas (por confirmar, el audio es ambiguo):** registro del póster antes del 16 de octubre; workshop/póster hacia el 18–20 de noviembre; informe final el 30; una frase sin fiabilidad menciona "16 de diciembre". Confirmar en el aula virtual.
- **Tema del grupo (deforestación amazónica con Sentinel-2):** entrenar con Brasil/Europa y probar en Perú (zero-shot). El archivo `4. Tarea Académica/datasets_proyecto_deforestacion.md` reúne datasets: MultiEarth 2023 (Pará, 17 215 recortes, S1/S2/Landsat), Bragagnolo (Zenodo 4498086, bosque/no bosque, CC BY 4.0), McCallum 2023 (43 108 imágenes S2), etiquetas Hansen GFC, JRC TMF, MapBiomas Perú, GLAD-S2 y RADD, y EuroSAT para pruebas rápidas. **Limitación:** Zenodo devolvió HTTP 403, no se descargó nada. **Decisiones pendientes:** cómo obtener las muestras, separar efectos de región, de etiquetado y de estacionalidad, DINOv2 o DINOv3, y aprobación del profesor porque el tema está fuera del banco.
- **Vibe Quest (cada clase, para la siguiente):** elegir una pregunta, investigar con IA, entregar un notebook interactivo (sliders), presentarlo en máximo 2 min al inicio de la siguiente clase y subirlo a la tarea "Vibe Quest". Si no puedes explicar de dónde salió algo, el Vibe Quest se cancela: "ustedes son los dueños del código". Preguntas de S9: espacio latente bayesiano vs frecuentista; pérdida con principios bayesianos y VAE; incertidumbre en el latente; manifold learning y PCA; regularizar el latente con propiedades del dominio.
- **Proyecto con Hugging Face (anunciado en S3):** elegir una aplicación de la plataforma (clasificación de texto o imagen, generación, multimodal, etc.), seleccionar el modelo base con su referencia bibliográfica, hacer ajuste de hiperparámetros (*fine-tuning*), proponer mejoras e implementarlas. Objetivo: aprender a usar modelos existentes y notar los límites de cómputo ("aquí falta hierro, necesito GPU"). Se invita a empezar leyendo papers. **Fechas dichas en S3 (por confirmar, parecen de otro semestre):** fin de semestre "primera semana de junio" y entrega "el 17 de junio"; no coinciden con el calendario 2026-2 de las otras clases. Quedan además el examen final y la propuesta de fechas que el profesor iba a confirmar en la clase siguiente.
- **Tarea de S4:** poner un regularizador, BatchNorm y Adam en el notebook de optimización (en Python/TensorFlow o ambos); plazo "el 14 a las 19:00" (**por confirmar**).

---

## 7. Pendientes

- [ ] Confirmar fechas del proyecto (registro del póster, workshop, informe; en S3 se dijo "17 de junio", que no cuadra con 2026-2) y la nota/rúbrica del parcial.
- [ ] Corregir en el notebook entregado, si el profesor lo permite: la frase de P2 (c) que dice que la `val_loss` de GAP "aún bajaba" en la época 10 (la curva muestra 0,585 en la época 9 y 0,61 en la 10). El código entregado no se modifica.
- [ ] No citar la cifra "0,867 con 30 épocas" del notebook (P5): no hay evidencia de esa corrida en la carpeta.
- [ ] `S2/1INF52_Lab01_Parte2.ipynb` y `S4/1INF52_Lab02.ipynb` siguen sin resolver.
- [ ] Verificar la hipótesis de la residual (añadir BatchNorm antes del Flatten) como práctica, sin cambiar lo entregado.
- [ ] Definir el tema del proyecto con el grupo y resolver la descarga de datos.
- [ ] Elegir una pregunta del Vibe Quest de S9 y preparar el notebook interactivo.

## 8. Plan sugerido (a partir de ahora)

1. Cerrar logística: fechas del proyecto y registro del póster.
2. Repasar las secciones 4 y 5 con ejemplos numéricos (parámetros, tamaños, IoU, entropía).
3. Practicar la explicación oral del examen (comprensión, no ejecución): tabla de ablación, Grad-CAM y diagnóstico.
4. Avanzar el proyecto con la regla "entender por qué" y preparar el Vibe Quest.
