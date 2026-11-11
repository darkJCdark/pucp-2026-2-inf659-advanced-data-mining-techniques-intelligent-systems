# Guion de 5 minutos · Vibe Quest S13 (pregunta 2)

## 0:00 – 0:40 · La pregunta
"La Clase 12 dice que la self-attention no distingue el orden y que por eso se suma el Positional Encoding (lámina 120). Yo lo pongo a prueba: dos frases con las mismas palabras y distinto orden, 'El gato persiguió al perro' y 'El perro persiguió al gato'. ¿Qué cambia en la salida si quito el PE?"

## 0:40 – 1:30 · El montaje
Notebook, secciones 3 y 4. Un codificador Transformer mínimo en PyTorch: self-attention, Add & Norm y feed-forward. Pesos aleatorios con semilla fija; no entreno nada porque la pregunta es sobre una propiedad de la arquitectura. Cada palabra tiene un solo embedding, esté donde esté. Mido el **error de equivarianza**: cuánto difiere la salida de B de la salida de A con las filas reordenadas.

## 1:30 – 2:30 · El resultado (sección 5 y gráficos 6 a 8)
- Sin PE: error 1.1e-15 y el "gato" tiene el mismo vector en las dos frases (coseno 1.0). Las matrices de atención de A y B son la misma con filas y columnas reordenadas.
- Con PE: error 1.47 y coseno 0.87. Las matrices cambian.
- En el PCA (sección 7), sin PE el círculo (A) y la cruz (B) de cada palabra se superponen; con PE se separan.

## 2:30 – 3:40 · Sliders (sección 9)
1. Dejo d = 16, 2 capas, escala 1 y **desmarco "usar PE"**: error ~1e-15.
2. Marco "usar PE" y subo capas a 6: el error sigue lejos de cero.
3. Subo la **escala del embedding** a 10: el PE pesa menos y el error baja (con d = 16 y 2 capas: 0.17 contra 1.47 con escala 1). Mensaje: el PE solo sirve si su señal no queda tapada por el embedding.
4. Pongo la escala del PE en 0: vuelve al caso sin PE.

## 3:40 – 4:30 · Robustez y fuentes
- 64 configuraciones y 100 permutaciones aleatorias en 3 frases: sin PE el error nunca pasa de 3.6e-15.
- PyTorch coincide con una implementación independiente en NumPy (sección 12).
- Fuentes: Zhang et al. (ICASSP 2024) dicen que los Transformers son position-agnostic; Banđur et al. (TPAMI 2026), que la self-attention no tiene noción nativa del orden.

## 4:30 – 5:00 · Límites
"Mi resultado vale para un codificador bidireccional con pesos aleatorios. Con máscara causal, un modelo sin PE puede inferir posición (Hayakawa y Sato, ACL 2025), y en sentimiento un modelo sin PE rindió parecido a RoPE (Guo, 2024). Probé un solo PE y una semilla de pesos."

## Preguntas posibles
- **¿Por qué no entrenaste?** La pregunta es estructural: con pesos compartidos entre posiciones y softmax simétrico, la equivarianza se cumple para cualquier peso. Entrenar mediría rendimiento, no esta propiedad.
- **¿Y en un decoder?** La máscara causal introduce asimetría y deja inferir la posición (Hayakawa y Sato, 2025). No lo probé.
- **¿Por qué el error con PE no sube con la escala del embedding?** No es monótono (0.97, 1.10, 0.74, 0.39); solo con escala 10 queda claramente más bajo. Con una semilla no afirmo más.
- **¿Qué es el error de equivarianza?** `max |salida(B) − salida(A)[perm]|`; cero significa que cada palabra salió igual y solo cambió de fila.
