# Vibe Quest S14 · ViT vs CNN en CIFAR-10 según la cantidad de datos

**Pregunta 2:** implementar una comparación entre ViT y CNN en CIFAR-10 con subconjuntos de 100, 1000 y 10000 imágenes, visualizar mapas de atención frente a mapas de características y determinar experimentalmente cuándo conviene cada arquitectura.

## Diseño
- Modelos preentrenados y congelados + sonda lineal (regresión logística L2, λ por validación cruzada de 3 pliegues dentro del subconjunto).
- ViT-S/16 (`vit_small_patch16_224.augreg_in21k_ft_in1k`, 21,67 M parámetros sin cabeza, 4,60 GMAC) frente a ConvNeXt-T (`convnext_tiny.fb_in22k_ft_in1k`, 27,82 M, 4,45 GMAC).
- CIFAR-10 (Hugging Face `uoft-cs/cifar10`): subconjuntos balanceados de n = 30, 100, 300, 1000, 3000 y 10000 (se piden 100/1000/10000), 3 semillas, evaluación en las 10 000 imágenes de prueba.
- Caso más difícil: CIFAR-100 (`uoft-cs/cifar100`, `fine_label`) con n = 300, 1000, 3000, 10000.
- Intervalos de confianza: bootstrap sobre las imágenes de prueba (2000 remuestreos, diferencia pareada promediada entre semillas).
- Visualización: attention rollout, atención del [CLS] por capa y cabeza, PCA de mapas de características, magnitud de activación de la CNN, Grad-CAM con la sonda lineal y análisis de Fourier descriptivo. Paneles con controles deslizantes.

## Resultados medidos (exactitud, media de 3 semillas; rejilla de λ de 8 valores, 1e-6 a 10)
| n | ViT-S/16 | ConvNeXt-T | ViT − ConvNeXt (pp) | IC95 | semillas donde gana el ViT |
|---|---|---|---|---|---|
| 30 | 83.43 | 86.09 | −2.66 | [−3.20; −2.12] | 0 de 3 |
| 100 | 92.02 | 90.16 | +1.86 | [1.42; 2.33] | 3 de 3 |
| 300 | 93.23 | 92.28 | +0.94 | [0.50; 1.39] | 2 de 3 |
| 1000 | 94.97 | 93.72 | +1.25 | [0.85; 1.64] | 3 de 3 |
| 3000 | 95.58 | 94.39 | +1.19 | [0.82; 1.60] | 3 de 3 |
| 10000 | 96.11 | 95.09 | +1.02 | [0.64; 1.42] | 3 de 3 |

CIFAR-100: ViT mejor en todos los tamaños: +1.12 pp con n=300 (IC95 [0.53; 1.75], 2 de 3 semillas) y +2.22, +2.61 y +2.39 pp con n=1000, 3000 y 10000. Con la rejilla de λ original (1e-4 a 1) n=300 había salido como empate, así que ese veredicto depende de la penalización elegida.

Eficiencia (T4, 60000 imágenes): ViT-S/16 1085 img/s y 1212 MiB; ConvNeXt-T 620 img/s y 2606 MiB.

Análisis de Fourier (512 imágenes, ventana ±0.125π): entre el primer y el último bloque, Δ log amplitud en 1.0π cambia +4.22 en ConvNeXt-T y +0.40 en ViT-S/16. No reproduce el filtrado pasa-bajos del ViT descrito por Park y Kim (2022) y difiere de su procedimiento; se deja como descripción.

Panel de mapas (sección 8.5), revisado con los valores por defecto (imagen 0, un gato; capa 12; etapa 4; sonda con n=1000): ambos modelos predicen "gato". El rollout del ViT se reparte por la imagen y la atención del [CLS] en la última capa se concentra en pocos parches; el Grad-CAM del ViT rodea la cabeza y el cuerpo del animal, y el de ConvNeXt-T (7×7) se centra en la cabeza. Es una sola imagen y no se midió la calidad de los mapas, así que no es evidencia sobre cuál modelo "mira mejor".

## Conclusión y límites
Con backbone preentrenado y solo una capa lineal entrenada, el ViT-S/16 conviene desde ~100 ejemplos y ConvNeXt-T solo con muy pocos (~30) en 10 clases; el cruce está entre 30 y 100 (no se probaron tamaños intermedios). **No se probó entrenamiento desde cero** (donde la literatura de datos pequeños reporta lo contrario), las recetas de preentrenamiento difieren entre modelos y es un solo par de modelos. El bootstrap no incluye la variación del subconjunto de entrenamiento (por eso se reporta en cuántas de las 3 semillas gana cada modelo). El 27 % de las sondas eligió una λ en el borde de la rejilla (casi siempre 1e-6, equivalente a no penalizar), y con 3 imágenes por clase esa elección es inestable.

## Contenido
- `resultados/google colab/vibe_quest_s14_vit_vs_cnn_cifar10.ipynb`: cuaderno ejecutado en Colab (T4), con salidas. Enlace: https://colab.research.google.com/drive/1KZ8gODaKLvWooJAdGtaPXcNUq9wtVqf0
- `codigo/vibe_quest_s14_vit_vs_cnn_cifar10.py`: exportación del cuaderno a script (generada por Colab).
- `resultados/datos/`: CSV de todas las tablas (exactitud en fracción 0–1; diferencias del bootstrap en puntos porcentuales).
- `resultados/capturas/`: figuras del cuaderno (curvas, diferencias con IC95, ejemplos de CIFAR-10 y espectro de Fourier). Los paneles con sliders solo se ven ejecutando el cuaderno.
- `pipeline.md`, `guion_5min.md`, `fuentes/` (PDF, `fuentes_doi.csv`, `README.md`).
