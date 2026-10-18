# Validación del notebook PyTorch en Google Colab — interpretación

Notebook ejecutado: `VibeQuest_S9_ManifoldLearning_PCA_pytorch.ipynb` (Colab, GPU T4, semilla 0, 1 500 puntos de rollo suizo, 80/20 train/test, error = MSE de reconstrucción en test).

## Qué se hizo y cómo
1. Se revisó la ejecución por defecto (celda 5: tabla de error con ruido 0 y aserción PCA ≈ AE lineal). La aserción pasó ("Verificado").
2. Se comprobó que los controles del Panel A responden: el botón AE lineal / AE no lineal y el deslizador de k cambian la figura (capturas 01–04). La figura se actualiza con varios segundos de retraso en el panel pequeño del navegador.
3. Para leer los errores con exactitud (la imagen de la barra es muy pequeña para leer decimales) se añadió **una celda extra al final de ese Colab** (no está en el `.ipynb` entregado) que recorre las 12 combinaciones k × ruido y llama a las mismas funciones `datos`, `reconstruir` y `error` sobre los modelos ya entrenados. La salida está en `datos/resultados_panelA.csv` (capturas 05–06 como evidencia).
4. También se ejecutó en una celda extra el bucle `comparar(k, ruido, tipo)` para las 24 combinaciones; la galería (captura 07) es solo parcial porque el visor del panel muestra cada figura demasiado pequeña.

## Resultados (Panel A)
| ruido | k | PCA | AE lineal | AE no lineal | mejora AE no lineal vs PCA |
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

## Qué pasa al mover cada control y por qué tiene sentido
- **k (1 → 2 → 3):** el error baja al subir k en los tres métodos. Con k = 3 el error es ≈ 0 porque el rollo ya es 3D: no se comprime nada. Con k = 1 la reconstrucción de PCA es una recta; con k = 2 es un plano. Esto coincide con la teoría de PCA (S9): se conserva la varianza de las primeras k direcciones.
- **AE lineal vs PCA:** coinciden hasta la 4.ª cifra decimal en las 8 combinaciones con k < 3 (diferencia máxima 0.06 %). Es el resultado de Baldi y Hornik (1989): un autoencoder lineal entrenado con MSE aprende el mismo subespacio que PCA. Que el entrenamiento con Adam llegue a lo mismo que la solución cerrada de PCA valida la implementación.
- **AE no lineal:** con k = 1 y k = 2 reconstruye mejor que PCA en todos los niveles de ruido. La razón es que las capas tanh permiten que la superficie de reconstrucción se doble y siga el rollo, algo que un plano no puede hacer. La ventaja es mayor con k = 2 (65–82 %) que con k = 1 (36–47 %), porque una superficie 2D curva puede aproximar mucho mejor un manifold 2D que una curva 1D.
- **Ruido (0 → 0.6):** el error sube en todos los métodos y la ventaja relativa del no lineal baja (82 % → 65 % en k = 2). Con ruido, parte de la varianza ya no está en el manifold sino en direcciones aleatorias que ningún método de dimensión baja reconstruye; el error irreducible crece y se reduce la distancia entre métodos.
- **k = 3 con AE no lineal:** el 0.0001 en lugar de 0.0000 es solo el error residual de optimización de la red (no es exactamente la identidad).

## Cómo responde a la pregunta del Vibe Quest
*Manifold learning en autoencoders y su relación con PCA.* Teórica: PCA es la mejor proyección lineal (el autoencoder lineal recupera ese subespacio); el autoencoder no lineal generaliza la idea a una variedad curva porque compone transformaciones con activaciones. Práctica: con datos que viven en una superficie curva, el no lineal reconstruye mejor con la misma dimensión latente, a costa de entrenar una red, ajustar hiperparámetros y perder la solución única y rápida de PCA.

## Límites que hay que decir al exponer
- **Una semilla no basta, pero el experimento de 10 semillas en PyTorch (ejecutado en Colab) lo confirma:** el AE no lineal ganó a PCA en 10 de 10 semillas en las 6 configuraciones (k = 1, 2 × ruido 0.0, 0.3, 0.6). Mejora media: k = 1 → 51 / 47 / 37 %; k = 2 → 79 / 68 / 57 %; mínimo 32 %. Detalle en `datos/resultados_semillas_pytorch_resumen.csv`.
- **El AE lineal no siempre iguala a PCA tan exactamente como con la semilla 0:** en las 10 semillas la diferencia máxima fue 0.4–1 % en la mayoría de configuraciones, pero llegó a **7.5 %** en una corrida con k = 1 y ruido 0.6 (la media del AE lineal en esa celda, 0.895, queda 1 % por encima de PCA, 0.885). Es un entrenamiento que no terminó de converger; por eso la tolerancia del `assert` del notebook es 10 %.
- **Mejor reconstrucción no significa "desenrollar" el rollo:** en la celda 5c la correlación del latente con la posición t en el rollo es baja (0.20 a 0.25 con PCA y autoencoders; 1.00 con Isomap sin ruido). No se debe afirmar que el autoencoder "aprende el manifold desenrollado", sino que reconstruye mejor.
- Los errores del CSV del Panel A son los impresos por la celda extra (precisión de 4 decimales). Las capturas del Panel A (01–04) muestran ruido 0.0 y k = 1, 2, 3; el Panel B se probó con método, ruido y latente 1 (capturas 11–18). No se capturaron las 24 combinaciones del Panel A por el retraso del visor; los números de las 24 están en el CSV.
