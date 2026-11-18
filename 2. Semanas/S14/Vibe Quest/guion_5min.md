# Guion · 5 minutos

**0:00–0:40 · Pregunta.** ¿Cuándo conviene un ViT y cuándo una CNN según cuántas imágenes etiquetadas hay? Probamos con 100, 1000 y 10000 imágenes de CIFAR-10 (y 30, 300, 3000 para ver la curva).

**0:40–1:30 · Diseño.** Dos modelos preentrenados y congelados, ViT-S/16 y ConvNeXt-T, de tamaño parecido. Solo se entrena una capa lineal con subconjuntos balanceados, 3 semillas, evaluando en las 10 000 imágenes de prueba. Declaro desde ya: no se entrenó desde cero.

**1:30–2:40 · Resultado.** Con 30 imágenes gana ConvNeXt (−2.66 pp, en las 3 semillas). Desde 100 gana el ViT: +1.86 pp con 100, +1.25 con 1000 y +1.02 con 10000, con intervalos del 95 % que no incluyen el cero. Con 300 la ventaja es la menor (+0.94) y gana en 2 de 3 semillas. La ventaja se reduce al crecer n.

**2:40–3:10 · Caso difícil.** En CIFAR-100 el ViT gana en todos los tamaños (+1.12 con 300; ≈ +2.4 desde 1000). Ojo: con otra rejilla de penalización el caso de 300 era un empate. El ViT además es más rápido (1085 vs 620 img/s) y usa menos memoria.

**3:10–4:10 · Mapas.** Demostrar el panel con sliders: rollout y atención del [CLS] por capa y cabeza en el ViT; PCA y magnitud de activación por etapa en la CNN; Grad-CAM de ambos. Son cualitativos, no prueban causalidad. En Fourier, el ViT no reproduce el pasa-bajos de Park y Kim.

**4:10–5:00 · Límites y conclusión.** Con backbone preentrenado y sonda lineal, ViT desde ~100 ejemplos. No se puede extrapolar a entrenamiento desde cero, hay confusión de receta de preentrenamiento y es un solo par de modelos.
