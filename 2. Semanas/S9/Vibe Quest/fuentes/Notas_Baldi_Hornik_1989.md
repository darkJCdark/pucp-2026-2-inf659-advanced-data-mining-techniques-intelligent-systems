# Notas de lectura — Baldi y Hornik (1989)

*Neural networks and principal component analysis: Learning from examples without local minima.* Neural Networks 2(1), 53–58.

Lectura: texto completo vía alphaXiv (PDF escaneado, OCR). Notas en mis palabras; no hay números de página verificados.

## Qué estudia
Una red de tres capas (entrada, una capa oculta más pequeña, salida) con unidades lineales, entrenada para reproducir su propia entrada minimizando el error cuadrático medio. Es el autoencoder lineal.

## Qué demuestran
1. La superficie de error tiene un único mínimo global; todos los demás puntos críticos son puntos de silla. No hay mínimos locales, así que el descenso del gradiente no se queda atascado en una solución mala.
2. En el mínimo, la capa oculta genera el mismo subespacio que las primeras k componentes principales de PCA.
3. Las matrices de pesos no son únicas: si A es el codificador y B el decodificador, cualquier par (A·C⁻¹, C·B) con C invertible da el mismo resultado. Por eso los ejes del latente no coinciden con los componentes principales ni vienen ordenados, aunque el subespacio sea el mismo.

## Cómo se usa en el notebook
- Es la base de la verificación PCA ≈ AE lineal (celda 5, `assert`).
- Explica por qué el AE lineal no puede superar a PCA y por qué el latente "se ve distinto" entre semillas (punto 3).
- Límite: el resultado vale para el mínimo global. Un entrenamiento con Adam y épocas fijas puede quedar cerca pero no exactamente en él (en 10 semillas, la diferencia llegó a 7.5 % en una corrida).

## Para decir en la exposición
"Sin activaciones, la red solo puede proyectar linealmente; Baldi y Hornik demostraron que lo mejor que puede hacer es el subespacio de PCA, y que el entrenamiento no tiene mínimos locales."
