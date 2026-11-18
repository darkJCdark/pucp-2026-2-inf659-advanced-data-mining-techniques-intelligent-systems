# Guion de 5 minutos · Vibe Quest S13 (pregunta 2)

Idea central: **sin PE, el Transformer ve una bolsa de palabras**. Todo lo demás es respaldo.

## 0:00 – 0:30 · La pregunta
"Si quito el Positional Encoding y le doy al Transformer 'El gato persiguió al perro' y 'El perro persiguió al gato', ¿las distingue? La Clase 12 dice que la self-attention no ve el orden (lámina 120). Lo compruebo comparando los embeddings de salida."

## 0:30 – 2:00 · La demostración (Parte A, secciones 2.5 a 2.8)
- Mismo modelo, mismos pesos, las dos frases. Sin PE: el "gato" sale con el mismo vector en ambas (coseno 1.0) y el error de equivarianza es 1e-15. Con PE: coseno 0.87 y error 1.47.
- Matrices de atención (2.7): sin PE, la de B es la de A con filas y columnas reordenadas.
- PCA (2.8): sin PE, el círculo y la cruz de cada palabra caen en el mismo punto. Con PE se separan.
- Por qué: Wq, Wk y Wv son iguales en toda posición y el softmax trata a todas por igual. Permutar la entrada solo permuta la salida.

## 2:00 – 2:40 · Slider (2.10)
Desmarco "usar PE": el error baja a ~1e-15. Lo marco: 1.47. Subo la escala del embedding a 10: el PE queda tapado y el error baja.

## 2:40 – 3:50 · ¿Se mantiene con un modelo entrenado? (sección 3.18)
- Entrené Transformers pequeños con oraciones reales de Tatoeba (español) y les pasé las dos frases de la consigna.
- Sin PE: el "gato" sale con el mismo vector en ambas frases (coseno 1.0, error 6e-7) y el modelo da la **misma** probabilidad a las dos (0.462). Para el modelo son la misma frase.
- Con PE: el coseno baja a 0.81 y el modelo ya las separa (0.567 contra 0.469).
- Muestro los cuatro PCA de la 3.18: sin PE, el círculo y la cruz de cada palabra coinciden; con PE se separan.

## 3:50 – 4:30 · Un matiz: la máscara causal
Un modelo con máscara causal sin PE sí distingue las frases (en la 3.18 su coseno de "gato" es 0.44): cada palabra ve un número distinto de palabras anteriores, y eso ya es posición (Zhang 2024; Hayakawa y Sato 2025). La respuesta vale para el codificador bidireccional de la clase.

## 4:30 – 5:00 · Conclusión y límites
"Sin PE, el codificador no sabe quién persigue a quién. El PE es lo que escribe el orden en cada vector. Límites: los modelos entrenados son pequeños, usé un solo tipo de PE y 3 semillas."

## Preguntas posibles
- **¿Y cuánto aprende el modelo con PE en general?** (respaldo: sección 3.12) En 2 000 oraciones con dos palabras intercambiadas: sin PE, 0.51 (azar); con PE, 0.59; si le quito el PE al evaluar, 0.50.
- **¿Por qué con PE solo llega a 0.59?** Detectar dos palabras intercambiadas es difícil para un modelo de d = 64 y 2 capas. Probé 25 épocas y leer el token `<BOS>` en vez del promedio, y ninguno mejoró. Lo que importa es la distancia al azar y a "sin PE".
- **¿Y si la tarea no depende del orden?** En identificar el idioma todos llegan a 0.995 con o sin PE. Ahí el PE no hace falta, aunque esa tarea es tan fácil que no deja margen para notar diferencias.
- **¿La Parte A no es obvia por construcción?** Sí: la invarianza sale de cómo está hecha la self-attention. Por eso la sección 3.18 lo comprueba con modelos entrenados.
- **¿Qué es el error de equivarianza?** `max |salida(B) − salida(A)[perm]|`: cero significa que cada palabra salió igual y solo cambió de fila.
