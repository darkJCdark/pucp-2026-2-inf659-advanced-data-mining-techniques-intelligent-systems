# Interpretación de resultados · P2 (Transformer sin Positional Encoding)

## Qué respondí
Pregunta: ¿qué pasa si quito el Positional Encoding y paso "El gato persiguió al perro" y "El perro persiguió al gato"? Resultado: **sin PE el codificador no distingue las dos frases; con PE sí.**

## Los números (PyTorch en Colab, igual a la referencia NumPy)
- **Frases de la consigna (d = 16, 2 capas):** sin PE, el error de equivarianza vale 1.1e-15 y el coseno entre el "gato" de A y el de B es 1.0. Con PE, el error vale 1.474 y el coseno baja a 0.869.
- **Barrido de 64 configuraciones** (d ∈ {8, 16, 32, 64}, 1 a 6 capas, escala del embedding 0.1 a 10): sin PE el error nunca pasa de 3.6e-15; con PE va de 0.049 a 2.283 (media 0.80).
- **100 permutaciones aleatorias en 3 frases** (5, 7 y 6 palabras, una con "gato" repetido): sin PE el error máximo es 2.7e-15; con PE la media está entre 1.20 y 1.31.

## Por qué
`Wq`, `Wk`, `Wv` son los mismos para toda posición y el softmax suma sobre todas las posiciones por igual. Permutar la entrada solo permuta las filas y columnas de la matriz de atención. El orden solo entra al modelo si algo lo escribe en el vector de cada token; eso hace el PE (lámina 120 de la Clase 12).

## Lo que no debo afirmar
- El experimento usa pesos aleatorios sin entrenar. Muestra una propiedad estructural, no cuánto empeoraría una tarea entrenada.
- Un decoder con máscara causal sin PE puede inferir la posición (Hayakawa y Sato, 2025), y en configuración causal el PE ayuda poco (Zhang et al., 2024). Mi resultado vale para el codificador bidireccional.
- En análisis de sentimiento, un modelo sin PE rindió parecido a RoPE (Guo, 2024): perder el orden no arruina toda tarea.
- Con PE, el error **no crece de forma monótona** con la escala del embedding (medias por escala: 0.97, 1.10, 0.74 y 0.39 para 0.1, 1, 3 y 10). Solo con escala 10 queda claramente más bajo; no puedo decir más con una semilla de pesos.
- Probé un solo tipo de PE (sinusoidal) y una semilla de pesos (0); la referencia NumPy y PyTorch dan lo mismo, pero no evalué otras semillas.
