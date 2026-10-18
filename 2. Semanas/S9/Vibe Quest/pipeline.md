# Pipeline del Vibe Quest S9 · Manifold learning y PCA

Documento de trabajo: qué se hizo, cómo, con qué resultados y qué no se pudo verificar. Se actualiza cada vez que cambia algo.

Curso: INF659 Técnicas Avanzadas de Data Mining y Sistemas Inteligentes (PUCP, 2026-2). Responsable: Jorge Chamorro.

## Historial

| Fecha | Cambio |
|---|---|
| 17 oct 2026 | Prototipo con sklearn (`MLPRegressor`), una semilla; plan de fuentes (IEEE, Scopus). |
| 18 oct 2026 | Versión PyTorch del notebook (la de clase), experimento de 10 semillas, validación en Colab (Panel A completo, Panel B representativo) y validación contra las fuentes. |
| 18 oct 2026 | Celdas 5b (tiempo) y 5c (Isomap); lectura completa de Baldi y Hornik y de Bengio; guion de 5 minutos. |
| 18 oct 2026 (versión final) | Reorganización de la carpeta. Se descarta el prototipo sklearn (era solo un paso previo; las cifras que valen son las de PyTorch). 5b promedia 20 ajustes de PCA para estabilizar el tiempo. Se actualiza la lista de limitaciones del notebook. Ejecución limpia del notebook final en Colab sin errores. |
| 18 oct 2026 (texto del notebook) | Se agregan celdas de texto (markdown) que explican cada paso, se reescriben los comentarios con lenguaje más natural y se redacta en primera persona cómo se trabajó (Jorge definió el planteamiento; la IA apoyó con fuentes, código y pruebas). El código no cambió: se comparó línea por línea sin comentarios (190 líneas iguales). |

## 1. Qué pide el Vibe Quest (fuente: `DM S9.txt`)

- Elegir **una** pregunta de las cinco y trabajarla con un asistente de IA.
- Entregar un **notebook interactivo** (con sliders) y subirlo a la tarea "Vibe Quest".
- Presentarlo al inicio de la siguiente clase. Duración: **5 minutos** (dato dado por Jorge; el audio de clase decía 2).
- Regla clave: si no se puede explicar de dónde salió algo, el Vibe Quest se cancela.

Pregunta elegida (la 4): *Explora el concepto de "manifold learning" en autoencoders y cómo se relaciona con PCA. ¿Cuáles son las implicaciones teóricas y prácticas de estas diferencias?*

Por qué esta pregunta: S9 cubrió PCA y el autoencoder lineal equivalente a PCA; se verifica con números (error de reconstrucción); se explica en lenguaje simple ("PCA proyecta sobre un plano, el autoencoder no lineal puede curvarse"); las preguntas 1, 2, 3 y 5 exigen inferencia bayesiana y VAE, vistos solo como concepto.

## 2. Contenido de la carpeta

| Ruta | Contenido |
|---|---|
| `VibeQuest_S9_ManifoldLearning_PCA.ipynb` | **Entregable.** Notebook PyTorch con sliders. |
| `Guion_VibeQuest_5min.md` | Guion, cifras verificadas y 15 preguntas probables. |
| `pipeline.md` | Este documento. |
| `codigo/experimento_semillas_pytorch.py` | Experimento con 10 semillas (se ejecuta aparte; no pegar en el mismo kernel del notebook). |
| `fuentes/` | PDF disponibles, notas de lectura y enlaces de descarga. |
| `resultados/` | Salidas de Google Colab: `capturas/`, `datos/` (CSV) e `interpretacion_resultados.md`. |
| `resultados/validacion_fuentes.md` | Contraste del notebook con las fuentes y el dataset. |

## 3. Diseño del experimento

Decisiones tomadas con Jorge:

- **Datos:** rollo suizo 3D sintético (`make_swiss_roll`, 1 500 puntos, estandarizado, división 80/20). Sin descargas; se ve a simple vista la diferencia entre un plano y una superficie curva. Es una elección didáctica: no viene de las fuentes.
- **Ruido:** patrón fijo escalado por 0.0, 0.15, 0.3, 0.6.
- **Dimensión latente:** k = 1, 2, 3 (k = 3 no comprime, sirve de control).
- **Métricas:** error cuadrático medio de reconstrucción en test; tiempo de ajuste; correlación de Spearman del mejor eje latente con t y confiabilidad (trustworthiness, 10 vecinos).
- **Entrenamiento (PyTorch):** Adam, tasa 0.005 con `ExponentialLR(0.9995)`, 2 500 épocas, lote completo. Autoencoder no lineal 3→32→k→32→3 con `tanh`; lineal 3→k→3 sin activaciones.

## 4. Celdas del notebook

Cada celda de código va precedida de una celda de texto que explica qué hace y por qué (22 celdas en total: 12 de texto y 10 de código).

| Celda | Contenido |
|---|---|
| 0 | Título, pregunta, cómo se trabajó (declaración de uso de IA) e idea central |
| 1 | Librerías y parámetros (semilla, niveles de ruido, valores de k) |
| 2 | Datos: rollo suizo, ruido fijo, división 80/20 |
| 3 | Clases `ModeloPCA` y `Autoencoder` con la misma interfaz `encode`/`decode` |
| 4 | Entrenamiento previo de 36 modelos (4 ruidos × 3 k × 3 métodos); ~90–110 s en Colab |
| 5 | Tabla de error y gráfico; `assert` PCA ≈ AE lineal (tolerancia 10 %) |
| 5b | Tiempo de ajuste PCA frente a autoencoders |
| 5c | Comparación con Isomap (Spearman y confiabilidad) |
| 6 | Panel A: reconstrucción 3D y error según k, ruido y tipo de autoencoder |
| 7 | Panel B: recorrido del espacio latente (k = 2) |
| 8 | Resumen por nivel de ruido; limitaciones y referencias en texto |

Conceptos del curso: PCA como proyección lineal (S9); autoencoder = codificador + cuello de botella + decodificador (S9); AE lineal equivale a PCA (S9; Baldi y Hornik, 1989); composición de lineales sigue siendo lineal y funciones de activación (S3); Adam y tasa de aprendizaje (S4); separar entrenamiento y prueba y estandarizar (S1–S4).

## 5. Resultados (todos ejecutados en Google Colab, GPU T4, redes en CPU)

### 5.1 Error de reconstrucción en test, semilla 0 (Panel A)

| Ruido | k | PCA | AE lineal | AE no lineal | Mejora del no lineal |
|---|---|---|---|---|---|
| 0.0 | 1 | 0.6239 | 0.6243 | 0.3291 | 47 % |
| 0.0 | 2 | 0.3049 | 0.3049 | 0.0537 | 82 % |
| 0.15 | 1 | 0.6429 | 0.6431 | 0.3473 | 46 % |
| 0.15 | 2 | 0.3178 | 0.3178 | 0.0655 | 79 % |
| 0.3 | 1 | 0.6924 | 0.6925 | 0.4014 | 42 % |
| 0.3 | 2 | 0.3466 | 0.3466 | 0.1160 | 67 % |
| 0.6 | 1 | 0.8822 | 0.8822 | 0.5632 | 36 % |
| 0.6 | 2 | 0.4519 | 0.4519 | 0.1584 | 65 % |
| cualquiera | 3 | 0.0000 | 0.0000 | ≤ 0.0001 | no aplica |

### 5.2 Diez semillas (cambian rollo, ruido, división e inicialización)

| k | Ruido | PCA | AE lineal | AE no lineal | Mejora media | Mejora mínima | Gana el no lineal |
|---|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.637 | 0.637 | 0.310 | 51.3 % | 41.6 % | 10/10 |
| 1 | 0.3 | 0.699 | 0.699 | 0.373 | 46.5 % | 42.0 % | 10/10 |
| 1 | 0.6 | 0.885 | 0.895 | 0.555 | 37.3 % | 32.3 % | 10/10 |
| 2 | 0.0 | 0.311 | 0.311 | 0.066 | 78.7 % | 68.4 % | 10/10 |
| 2 | 0.3 | 0.337 | 0.337 | 0.107 | 68.4 % | 60.2 % | 10/10 |
| 2 | 0.6 | 0.418 | 0.417 | 0.177 | 57.2 % | 47.2 % | 10/10 |

El AE lineal difiere de PCA 0.4–1 % en la mayoría de los casos y llegó a **7.5 %** en una corrida (k = 1, ruido 0.6): el entrenamiento no siempre alcanza el óptimo teórico. Los valores se leyeron de capturas de pantalla (3 decimales); el CSV bruto de esa ejecución no se guardó (`resultados/datos/resultados_semillas_pytorch_resumen.csv` es el resumen).

### 5.3 Tiempo e Isomap (k = 2)

| Medida | PCA | AE lineal | AE no lineal | Isomap |
|---|---|---|---|---|
| Tiempo de ajuste | milisegundos | 2–5 s | 5–6 s | ~0.6 s |
| Spearman con t, ruido 0 | 0.25 | 0.23 | 0.20 | 1.00 |
| Confiabilidad, ruido 0 | 0.860 | 0.862 | 0.973 | 0.999 |
| Spearman con t, ruido 0.3 | 0.22 | 0.19 | 0.25 | 0.20 |
| Confiabilidad, ruido 0.3 | 0.843 | 0.844 | 0.961 | 0.878 |

Los tiempos varían entre ejecuciones y máquinas (PCA 0.001–0.008 s; AE lineal 2.3–4.5 s; AE no lineal 4.9–6.0 s en las corridas de Colab). El cociente "veces PCA" va de cientos a miles: solo es estable el orden de magnitud. Isomap no tiene decodificador en sklearn, por eso no se mide su error de reconstrucción.

**Lectura:** el autoencoder no lineal reconstruye mejor, pero su latente no despliega el rollo (Spearman 0.20–0.25); Isomap sí lo despliega sin ruido y falla con ruido 0.3. El Panel B lo confirma visualmente: PCA y AE lineal dan un plano; el AE no lineal, una hoja curva; en los tres, los colores quedan mezclados en el espacio latente.

## 6. Fuentes

| Fuente | Estado de lectura | Uso |
|---|---|---|
| Baldi y Hornik (1989), Neural Networks 2(1) | **Leída completa** (alphaXiv); PDF en `fuentes/` solo si se descargó | Equivalencia AE lineal = PCA; sin mínimos locales; latente no único |
| Bengio, Courville y Vincent (2013), IEEE TPAMI 35(8) | **Leída completa** (alphaXiv) | Hipótesis del manifold; PCA = manifold lineal; Isomap sin función para puntos nuevos |
| Fournier y Aloise (2019), IEEE AIKE | Resumen y consultas puntuales; PDF en `fuentes/` | AE mejor con pocas dimensiones; PCA ~100× más rápido; recomiendan PCA para prototipar |
| Wetzel (2017), Phys. Rev. E 96(2) | Resumen y consultas puntuales; PDF en `fuentes/` | AE lineal ≈ PCA; PCA más preciso que AE y VAE en Ising |
| Hinton y Salakhutdinov (2006), Science 313 | Solo resumen | AE profundo frente a PCA |
| Jolliffe y Cadima (2016), Phil. Trans. R. Soc. A 374 | Solo resumen | Descripción rigurosa de PCA |
| Tenenbaum, de Silva y Langford (2000), Science 290 | No leída; el rollo suizo como ejemplo del Isomap se confirmó con apuntes de otro curso y la bibliografía de Fournier y Aloise | Referencia del dataset y de Isomap |

Verificación por DOI en Scopus (17 oct 2026). Un DOI supuesto para Alkhayrat et al. resultó pertenecer a otro artículo y esa fuente se descartó. Notas de lectura en `fuentes/`.

## 7. Limitaciones y riesgos (para la defensa)

1. **Una sola semilla en el notebook;** las 10 semillas están en `codigo/` y se ejecutaron aparte.
2. **Sin conjunto de validación.** Los hiperparámetros se eligieron con un prototipo con la semilla 0 y se mantuvieron en las 10 semillas.
3. **El latente no desenrolla el rollo.** Se defiende "reconstruye mejor", no "aprende el manifold desenrollado".
4. **El latente no es único;** otra semilla da otros ejes.
5. **Datos sintéticos de 3 dimensiones;** no se probó con imágenes reales ni se midió clasificación.
6. **El dataset no viene de las fuentes** (elección didáctica; clásico del Isomap).
7. **AE lineal ≈ PCA, no igual:** hasta 7.5 % de diferencia en una corrida.
8. **Tiempos:** orden de magnitud, no decimales.
9. **Interactividad:** en el panel pequeño de Colab la figura tarda varios segundos en actualizarse tras mover un slider.
10. **Contaminación de variables en Colab:** si en el mismo kernel se ejecuta el script de semillas por `exec`, sobrescribe `Autoencoder`, `NIVELES_RUIDO` y `t` y rompe las celdas siguientes. Ejecutar el script aparte.

## 8. Validación en Colab

- Ejecución limpia del notebook final (código de las 10 celdas de código, desde un espacio de nombres vacío): sin errores; `assert` "Verificado"; tablas 5, 5b, 5c y resumen impresos; cifras iguales a la sección 5.
- Incidentes en la validación (no del notebook): (a) mis celdas auxiliares en el Colab de Jorge reasignaron `Autoencoder`, `NIVELES_RUIDO` y `t` (limitación 10) y provocaron un `TypeError: ... missing 'semilla'` y un Spearman de Isomap falso (≈ 0.1); (b) causa comprobada con `np.allclose`. Con un espacio limpio no ocurre.
- Para repetirlo: subir el `.ipynb` a Colab → Entorno de ejecución → Ejecutar todo. No pegar celdas auxiliares.

## 9. Recomendaciones antes de exponer

1. Subir `VibeQuest_S9_ManifoldLearning_PCA.ipynb` a un Colab nuevo (el actual tiene celdas auxiliares y versión antigua) y ejecutar todo.
2. Revisar `Guion_VibeQuest_5min.md` y ensayar con cronómetro.
3. Descargar a `fuentes/` los PDF que falten (README de esa carpeta).
4. Subir el notebook a la tarea "Vibe Quest".
