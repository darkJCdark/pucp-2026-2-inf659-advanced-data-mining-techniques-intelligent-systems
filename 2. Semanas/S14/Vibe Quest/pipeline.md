# Pipeline S14

1. **Preparación**: instalar `timm` y `datasets`; GPU T4; parámetros globales (semillas 0–2, tamaños, λ, N_BOOT=2000).
2. **Datos**: CIFAR-10 desde Hugging Face (50 000 train / 10 000 test), imágenes 32→224 bicúbico.
3. **Modelos**: ViT-S/16 y ConvNeXt-T preentrenados (ImageNet-21k/22k → 1k), congelados; parámetros y GFLOPs medidos.
4. **Extracción**: rasgos de las 50 000 + 10 000 imágenes una sola vez (fp16, lote 256); se miden tiempo y memoria.
5. **Sonda lineal**: subconjuntos balanceados por semilla; estandarización con media/desv. del subconjunto; L-BFGS con L2; λ por CV de 3 pliegues en una rejilla de 8 valores (1e-6 a 10); evaluación en 10 000 de prueba (36 sondas).
6. **Bootstrap**: diferencia pareada ViT − ConvNeXt, IC95 por percentiles, más conteo de semillas donde gana cada modelo.
7. **CIFAR-100**: mismo procedimiento con n = 300–10000.
8. **Visualización**: rollout, atención por capa/cabeza, PCA, Grad-CAM con la sonda, espectro de Fourier; paneles con sliders.
9. **Chequeos**: balance de subconjuntos, parámetros, NaN, filas del rollout suman 1, Grad-CAM válido, y conteo de λ en el borde de la rejilla.
10. **Interpretación y límites**, **guardado** (CSV + PNG + zip) y **fuentes** APA 7.
