# Notas de lectura — Bengio, Courville y Vincent (2013)

*Representation learning: A review and new perspectives.* IEEE TPAMI 35(8), 1798–1828.

Lectura: texto completo vía alphaXiv. Notas en mis palabras; no hay números de página verificados (se cita por sección y nota al pie).

## Ideas usadas
1. **Hipótesis del manifold (sección 8):** los datos de alta dimensión se concentran cerca de una variedad de dimensión mucho menor. Aprender una representación es, en buena parte, aprender coordenadas sobre esa variedad.
2. **PCA como manifold lineal:** PCA asume que la variedad es un subespacio lineal; es el caso más simple de manifold learning.
3. **Métodos no paramétricos (Isomap, LLE, etc.):** calculan coordenadas para los puntos de entrenamiento, pero no entregan una función que se aplique directamente a puntos nuevos ni un decodificador. Los autoencoders sí aprenden esa función (codificador y decodificador).
4. **Nota al pie 15:** un autoencoder lineal entrenado con error cuadrático recupera el mismo subespacio que PCA, pero sus pesos no son ortonormales ni están ordenados por varianza (coincide con Baldi y Hornik).

## Cómo se usa en el notebook
- Marco conceptual de la pregunta: PCA = manifold lineal; autoencoder no lineal = manifold curvo parametrizado.
- Celda 5c: se compara con Isomap precisamente por el punto 3 (Isomap despliega el rollo, pero sin decodificador no mide error de reconstrucción).

## Límite
Bengio plantea la hipótesis del manifold como supuesto de trabajo, no como resultado demostrado para cualquier conjunto de datos.
