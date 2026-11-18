# Interpretación de resultados · P2 (Transformer sin Positional Encoding)

## Respuesta
Sin PE, el Transformer (codificador bidireccional) no distingue "El gato persiguió al perro" de "El perro persiguió al gato": cada palabra sale con el mismo vector en las dos frases y solo cambia de fila.

## Evidencia
- **Parte A, pesos aleatorios:** error de equivarianza 1.1e-15 sin PE y 1.47 con PE; coseno de "gato" 1.0 y 0.87. Igual en 64 configuraciones y 100 permutaciones de 3 frases (sin PE nunca pasa de 3.6e-15).
- **Parte B, modelos entrenados con Tatoeba (sección 3.18):** bidireccional sin PE, error 6e-7, coseno 1.0 y la misma P(original) para ambas frases (0.462); con PE, coseno 0.81 y 0.567 contra 0.469.
- **Tarea de orden (2 000 oraciones de prueba, 3 semillas):** bidireccional sin PE 0.508 ± 0.016 (azar); con PE 0.588 ± 0.008; con PE pero quitándolo al evaluar 0.499 ± 0.015.

## Por qué
Pesos compartidos entre posiciones y softmax simétrico: permutar la entrada permuta la salida. Antes de entrenar, el modelo de PyTorch da una diferencia de 2.4e-7 al permutar sin PE y 0.10 con PE (sección 3.9).

## Matices de las fuentes
- **Causal:** sin PE 0.626 ± 0.010 y con PE 0.623 ± 0.002. La máscara causal da posición (Zhang et al., 2024, p. 1004; Hayakawa y Sato, 2025).
- **Idioma:** 0.995-0.996 en todas las configuraciones (Guo, 2024). Quitar el PE a modelos entrenados con él baja a 0.956-0.962.

## Límites
Modelo pequeño (la tarea de orden llega a 0.59-0.63; con 25 épocas no mejora). Un tipo de PE, 3 semillas y un dataset. Parte A con pesos aleatorios.
