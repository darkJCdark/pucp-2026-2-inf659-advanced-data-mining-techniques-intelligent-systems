# Fuentes S14

Criterio acordado: Scopus/IEEE 2022–2026; excepciones marcadas: Selvaraju (2017, Grad-CAM) y Abnar y Zuidema (2020, rollout).

| Archivo | Uso en el trabajo | Estado de verificación |
|---|---|---|
| Liu_2022_ConvNeXt_CVPR.pdf | Diseño de ConvNeXt; la versión CVPR indica 29 M parámetros en la tabla 1 | Leído (secciones 2.1, 3.2) |
| Lu_2022_ViT_CNN_small_NeurIPS.pdf | ViT vs CNN en datos pequeños (CIFAR-100, DomainNet; no usa CIFAR-10) | Leído (pp. 1–2, 6–8) |
| Park_2022_How_do_ViT_work_ICLR.pdf | Atención como suavizado; análisis de Fourier (no muestra mapas de atención) | Leído (secciones 2 y 3) |
| Touvron_2022_Three_things_ViT_ECCV.pdf | Ajuste fino de ViT; efecto techo en CIFAR-10 | Leído parcialmente (p. 11) |
| Jelassi_2022_ViT_spatial_structure_NeurIPS.pdf | No se cita en el cuaderno (retirada tras la auditoría) | No usada |
| Wang_2022_Transferability_supervised_pretraining_CVPR.pdf | Transferibilidad del preentrenamiento supervisado; evaluación lineal sobre backbone fijo (Tabla 4, p. 8) | Leída la Tabla 4 |
| Zhang_2025_DepthWise_ViT_Neurocomputing.pdf | ViT en datos pequeños; DOI 10.1016/j.neucom.2024.128998 (verificado en el PDF) | Leído el resumen y la introducción |
| Rostand_2025_Locality_guidance_TNNLS.pdf | ViT en datasets diminutos; DOI 10.1109/TNNLS.2024.3515076 (verificado en el PDF) | Leído el resumen y la introducción |
| Selvaraju_2017_GradCAM_ICCV.pdf | Método Grad-CAM (excepción) | Método aplicado |
| Abnar_2020_Attention_rollout_ACL.pdf | Attention rollout (excepción) | Método aplicado |

Los DOI de las demás fuentes no se verificaron en el PDF: se citan con la URL oficial. Ver `fuentes_doi.csv` y la sección 12 del cuaderno.


Fuentes añadidas fuera de la carpeta (no son PDF descargados): Steiner et al. (2021, arXiv 2106.10270, pesos del ViT; excepción anterior a 2022), Wightman (2019, timm, DOI 10.5281/zenodo.4414861, tomado de la ficha del modelo) y Krizhevsky (2009, CIFAR). Fichas de Hugging Face revisadas el 2026-11-18.

Páginas verificadas para los métodos aplicados: Wang et al. (2022), apéndice H.3, p. 22 (protocolo de evaluación lineal); Abnar y Zuidema (2020), pp. 3–4 (A = 0.5·W + 0.5·I y producto de capas); Selvaraju et al. (2017), ec. 1, p. 4 (promedio global de gradientes); Park y Kim (2022), apéndice A.2, p. 16 (Δ log amplitud entre 1.0π y 0, ventana de 0.1π, todo el conjunto de prueba).
