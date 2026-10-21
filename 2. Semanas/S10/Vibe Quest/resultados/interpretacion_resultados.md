# Interpretación de resultados (Colab, semilla 0 salvo indicación)

Datos: `datos/resultados_*.csv`. Figuras: `capturas/fig1_*.png`, `fig2_*.png`.

## 1. Superposición (H1)
Con 4 neuronas y 12 rasgos, el modelo representa 8 (esparsidad 0.5), 11 (0.9) y 12 (0.97). La pureza de las neuronas es 0.55, 0.45 y 0.42. Más esparsidad, más rasgos comprimidos y neuronas más mezcladas.

## 2. Recuperación con SAE (H2)
- α = 0.3: recupera 11/11 y 12/12 con esparsidad 0.9 y 0.97 (pureza de latentes 0.85 y 0.89).
- α = 0: recupera 0.27 y 0.25. α = 1: el error de reconstrucción sube a ≈ 0.48.
- Esparsidad 0.5: máximo 0.375; el SAE no mejora la pureza (0.56 frente a 0.55).
- Cinco semillas (0.97, α = 0.3): recupera 0.80 ± 0.14 (rango 0.58 a 0.92). Cifra principal: 0.80, no 1.00.

## 3. Steering (H3)
Con factor 3: selectividad promedio SAE 0.81 ± 0.10 frente a 0.23 ± 0.03 de la neurona (5 semillas). En la semilla 0 con esparsidad 0.9 el SAE da 0.85 a 1.42 y la neurona 0.00 a 0.59. Selectividad menor que 1 en el mundo más esparso: hay efectos colaterales.

## Límites
Mundo sintético, una capa, hiperparámetros afinados a mano, métricas propias. No se infiere nada sobre modelos reales a partir de esto.
