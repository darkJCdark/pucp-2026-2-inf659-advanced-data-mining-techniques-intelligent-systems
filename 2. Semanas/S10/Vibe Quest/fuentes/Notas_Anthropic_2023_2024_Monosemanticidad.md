# Notas — Anthropic (2023, 2024): Towards Monosemanticity y Scaling Monosemanticity

Referencias:
- Bricken, T. et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Transformer Circuits Thread. https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Templeton, A. et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. Transformer Circuits Thread. https://transformer-circuits.pub/2024/scaling-monosemanticity/

Estado de lectura: páginas web leídas con resumen asistido; en cada una quedó sin leer la cola (≈22 % de *Scaling*, ≈28 % de *Towards*). No hay PDF oficial; se guarda una copia impresa de cada página si el navegador permite descargarla.
Notas en palabras propias.

## Towards Monosemanticity (2023)
- Modelo: transformer de una capa con MLP de 512 neuronas; SAE con ReLU, decodificador lineal, sesgos ligados, pérdida MSE + L1, Adam; caso principal con 4 096 rasgos (expansión 8×).
- En evaluación humana con rúbrica, los rasgos puntúan muy por encima de las neuronas; en la interpretación automática también ganan.
- Recuperan ≈79 % de la reducción de pérdida que aporta la capa MLP en el caso principal.
- Limitaciones: monosemanticidad no total (queda polisemia en activaciones bajas), sin métrica fiable para evaluar el diccionario, modelo de una sola capa.

## Scaling Monosemanticity (2024)
- Tres SAE sobre el flujo residual de la capa intermedia de Claude 3 Sonnet, con ≈1 M, 4 M y 34 M de rasgos; pérdida L2 + L1.
- En promedio, menos de 300 rasgos activos por token y al menos 65 % de varianza explicada; rasgos muertos: ≈2 % (1 M), ≈35 % (4 M), ≈65 % (34 M).
- Steering: fijar (clamp) un rasgo a un valor artificialmente alto durante el paso hacia adelante cambia el comportamiento. Ejemplo emblemático: el rasgo del puente Golden Gate, fijado a un múltiplo alto de su máximo (≈10×), hace que el modelo se identifique con el puente (es el caso de la lámina 47 de la clase 10).
- Rasgos relevantes para seguridad que reportan: código inseguro y puertas traseras, sesgos (p. ej. una de género en profesiones), adulación, engaño y manipulación, contenido peligroso.
- Advertencias de los autores: los hallazgos son preliminares; no demuestran que algún rasgo sea útil para seguridad; existir un rasgo no implica que el modelo "tenga" ese objetivo; hay riesgos de artefactos del aprendizaje de diccionario y de efectos aguas abajo que no coinciden con lo que sugiere la activación.
- Cobertura incompleta: p. ej. solo ≈60 % de los distritos de Londres tenía rasgo asociado en el SAE de 34 M.

## Cómo lo uso
- Panel B del cuaderno imita el *clamp*; la diferencia es que yo sé cuál es el rasgo verdadero.
- Las advertencias de los autores son la base de la sección de límites y de riesgos éticos del cuaderno.
