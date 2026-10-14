# Datasets para el proyecto grupal: deforestación amazónica con Sentinel-2 (entrenamiento Brasil/Europa, prueba Perú)

Curso INF659, proyecto grupal WAIMLAp 2026. Búsqueda del 13 de octubre de 2026.

**Convención de verificación.**
- **Verificado:** abrí la página o el paper y los datos de la ficha salen de ahí.
- **No confirmado:** la página no indica ese dato o no pude abrirla.
- No se descargó ningún dataset: Zenodo devolvió HTTP 403 desde el entorno en la nube y desde la computadora del usuario (lista de dominios permitidos). Todo lo de abajo es documental.

---

## 1. Entrenamiento: Sentinel-2 (u otros sensores) con etiquetas de deforestación o bosque

| Dataset | Enlace | Sensor / bandas | Etiquetas | Cobertura | Licencia | Notas |
|---|---|---|---|---|---|---|
| MultiEarth 2023 | [paper](https://arxiv.org/abs/2306.04738) · [GitHub](https://github.com/MIT-AI-Accelerator/multiearth-challenge) · [sitio del reto](https://sites.google.com/view/rainforest-challenge/multiearth-2023) | Sentinel-1 (VV, VH), Sentinel-2 (12 bandas, 10 m, 256×256), Landsat 5 y 8 (30 m) | Manuales sobre mosaicos Planet (3,7 m), 11 fechas 2016–2021, sin temporada de lluvias (oct–abr). 17.215 recortes de entrenamiento | Zona de Pará (Brasil), ~9.500 km² según [Lee & Choi 2023](https://www.mdpi.com/2072-4292/15/21/5167). **No incluye Perú** | Uso libre para desarrollo, investigación y educación (remite a los términos de Google Earth Engine) | Descarga en NetCDF y TIFF comprimidos desde el sitio del reto. El repo solo trae una muestra pequeña |
| Amazon and Atlantic Forest image datasets (Bragagnolo et al.) | [Zenodo 4498086](https://zenodo.org/records/4498086) · [AMAZON.rar](https://zenodo.org/records/4498086/files/AMAZON.rar?download=1) | Sentinel-2 L2A, bandas 4, 3, 2 y 8 convertidas a byte (0–255), GeoTIFF 512×512 | Máscara bosque / no bosque (PNG) | Amazonía y Mata Atlántica brasileñas | CC BY 4.0 | 2,3 GB. 499 imágenes de entrenamiento de Amazonía y 100 de validación. No es cambio, es bosque vs. no bosque |
| Amazon Rainforest dataset for semantic segmentation | [Zenodo 3233081](https://zenodo.org/records/3233081) | No se especifica en la página | Máscara bosque / no bosque (PNG) | Amazonía (sin coordenadas) | CC BY 4.0 | 21,8 MB. 30 imágenes de entrenamiento, 15 de validación, 15 de prueba. Solo para pruebas rápidas |
| Deforestation in Ukraine (Sentinel-2) | [Kaggle](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine) | Sentinel-2 | Deforestación | Ucrania | No confirmado | Listado en el catálogo satellite-image-deep-learning; no lo abrí |
| Borneo Forest Disturbance Dataset | [GitHub](https://github.com/ColmKeyes/Borneo_Forest_Disturbance_Dataset) | Sentinel-2 + alertas RADD | Perturbaciones forestales | Borneo | No confirmado | Listado en el catálogo; no lo abrí |
| Ghana Sentinel-2 Forest | [Kaggle](https://www.kaggle.com/datasets/kwabenaobeng/ghana-sentinel2-forest) | Sentinel-2 | Bosque | Ghana | No confirmado | Listado en el catálogo; no lo abrí |
| Amazon Rainforest Deforestation Dataset | [Kaggle](https://www.kaggle.com/datasets/akhilchibber/deforestation-detection-dataset) | No indicado | No indicadas | Amazonía | No indicada | Solo vi metadatos: "Deep Learning Based Multi-Source Data Fusion to Map Deforested Areas" |
| Muestras de McCallum et al. 2023 | [paper](https://doi.org/10.3390/rs15215204) | Sentinel-2 RGB, 10 m, mosaicos de 3×3 km (2018–2020) | Binarias (hay / no hay deforestación) por consenso de >5.500 voluntarios, 43.108 imágenes | Amazonía brasileña (entrenamiento) y peruana (validación) | No confirmada | El paper dice que hay ejemplos en Zenodo, pero no encontré la URL exacta |
| Roboflow Universe "deforestation" | [ejemplo](https://universe.roboflow.com/object-detection-lazxt/deforestation-yzbgm) | No confirmado que sea Sentinel-2 | Variable | Variable | Variable | Inspeccionar antes de usar. Muchos proyectos son detección de objetos sobre otro tipo de imagen |

## 2. Etiquetas y alertas para Perú (prueba zero-shot)

| Producto | Enlace | Resolución / sensor | Periodo | Licencia | Notas |
|---|---|---|---|---|---|
| Hansen Global Forest Change | [Earth Engine v1.12](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2024_v1_12) | 30,92 m, Landsat | 2000–2024 | CC-BY-4.0 | La versión v1.12 figura como obsoleta. La reemplaza `UMD/hansen/global_forest_change_2025_v1_13`. Bandas: `treecover2000`, `loss`, `lossyear`, `gain`, `datamask` |
| JRC Tropical Moist Forests (TMF) | [source.coop](https://source.coop/epoch/jrc-tmf) · [catálogo STAC](https://data.source.coop/epoch/jrc-tmf/catalog.json) | ~30 m, Landsat | 1990–2025 | Datos abiertos Copernicus/JRC, con atribución | COG, acceso anónimo. Cobertura pantropical. Perú no aparece nombrado: hay que filtrar el índice STAC por huella geográfica |
| MapBiomas Perú | [descargas](https://peru.mapbiomas.org/en/descargas) | Mapas por colección, mosaicos Landsat anuales desde 1985 | Según colección | CC BY 4.0 | Resolución y formato no especificados en la página |
| MapBiomas Alerta Perú | [plataforma](https://plataforma.peru.alerta.mapbiomas.org/mapa) | Alertas de deforestación validadas | No confirmado | No confirmada | No hay enlace de descarga directo en la página de descargas |
| GeoBosques / Minam | [descargas](https://geobosques.minam.gob.pe/geobosque/view/descargas.php) · [Programa Bosques](http://www.bosques.gob.pe/) | No confirmado | No confirmado | No confirmada | Pide un formulario de registro de usuario. No pude ver qué capas ni qué formato ofrece |
| GLAD-S2 | [blog](https://globalnaturewatch.org/blog/data-and-tools/glad-s2-offers-high-resolution-deforestation-alerts/) · [glad.earthengine.app](https://glad.earthengine.app) | Sentinel-2, 10 m | Alertas 2019 en adelante | No indicada | En 2021 cubría bosque primario húmedo de la cuenca amazónica. No verifiqué la cobertura actual |
| RADD | [catálogo comunitario de GEE](https://gee-community-catalog.org/projects/radd/) | Sentinel-1, 10 m | Última actualización marcada: 2023-01-08 | CC BY 4.0 | ID en Earth Engine: `projects/radar-wur/raddalert/v1`. Sudamérica con `geography = 'sa'` |
| Sentinel-1 for Science Amazonas | [catálogo ESA](https://opensciencedata.esa.int/projects/s14scienceamazonas/collection) | Sentinel-1 | No indicado | No indicada | Mapas de pérdida forestal con series SAR. No pude ver más detalles |
| Dynamic World | [Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) | Sentinel-2, 10 m, 9 clases | Desde 2015-06-27 | CC-BY 4.0 | ID: `GOOGLE/DYNAMICWORLD/V1`. Útil como máscara de bosque (clase "árboles") |
| ESA WorldCover | [v200](https://developers.google.cn/earth-engine/datasets/catalog/ESA_WorldCover_v200?hl=en) | Sentinel-2, 10 m | Mapas anuales (2020/2021) | No verificada | Solo apareció en la búsqueda; no abrí la ficha |
| Planet NICFI | [planet.com/nicfi](https://www.planet.com/nicfi/) | Mosaicos de alta resolución de los trópicos | No confirmado | No confirmada | Listado en el catálogo satellite-image-deep-learning. No verifiqué si el programa sigue vigente ni sus condiciones |

## 3. Europa, preentrenamiento y benchmarks

| Dataset | Enlace | Detalle | Licencia / tamaño |
|---|---|---|---|
| EuroSAT | [Zenodo 7711810](https://zenodo.org/records/7711810) · [RGB](https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1) · [multiespectral](https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1) · [GitHub](https://github.com/phelber/EuroSAT) · [paper](https://arxiv.org/abs/1709.00029) | ~27.000 imágenes de 64×64, 10 clases de uso de suelo, 13 bandas o RGB, Europa. Es el dataset de la captura de Kaggle (coincide por cifras; no abrí esa página) | MIT. RGB 94,7 MB, MS 2,1 GB |
| EuroSAT-SAR | [Hugging Face](https://huggingface.co/datasets/wangyi111/EuroSAT-SAR) | Pares Sentinel-1 para cada imagen de EuroSAT | No confirmada |
| SSL4EO-S12 downstream (Embed2Scale) | [Hugging Face](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream) | Datacubes de 27 canales (S1 + S2 L1C + S2 L2A), 4 estaciones. Conjunto de desarrollo: 5.149 datacubes | CC BY 4.0. ~60 GB (desarrollo) y ~90 GB (evaluación). Demasiado grande para Colab gratuito sin streaming |
| Major TOM | [Hugging Face](https://huggingface.co/Major-TOM) · [ESA Φ-lab](https://philab.esa.int/hello-major-tom) | Sentinel-2 global, cuadrícula fija de 10 km | ~50 TB. Licencia no indicada en el artículo. Solo con subconjuntos |
| BigEarthNet | [paper](https://arxiv.org/abs/1902.06148) | Benchmark multietiqueta Sentinel-2 | No verifiqué descarga ni tamaño |
| OSCD (Onera) | [sitio](https://rcdaudt.github.io/oscd/) · [versión RGB en HF](https://huggingface.co/datasets/blanchon/OSCD_RGB/tree/8fcfa87f59956ceedf20299e050f59f97bd15e8e) | 24 pares multiespectrales Sentinel-2 (2015–2018), cambio urbano | No confirmada |
| PANGAEA | [paper](https://arxiv.org/abs/2412.04204) | Benchmark de modelos fundacionales geoespaciales | — |
| GeoCrossBench | [paper](https://arxiv.org/abs/2511.02831) | Generalización entre bandas en teledetección | — |
| PhilEO downstream / pretrain | [downstream](https://huggingface.co/datasets/PhilEO-community/PhilEO-downstream) · [pretrain](https://huggingface.co/datasets/PhilEO-community/PhilEO-pretrain) | Densidad de edificios, carreteras y cobertura; preentrenamiento global | ~400 GB y ~500 GB (según el catálogo). No viable en Colab gratuito |

## 4. Clima y simulación (se mantiene como canales de entrada del modelo)

| Dataset | Enlace | Resolución | Periodo | Notas |
|---|---|---|---|---|
| ERA5-Land | [documentación ECMWF](https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation) | 0,1° en el CDS (nativo ~9 km), horario | Desde 1950 hasta ~2–3 meses atrás | Descarga con `cdsapi` (conjunto `reanalysis-era5-land`). Hay que citar la entrada del catálogo y atribuir a Copernicus. Ojo: ERA5 atmosférico (0,25°) es más grueso que ERA5-Land |
| PISCO precipitación (SENAMHI) | [repositorio SENAMHI](https://repositorio.senamhi.gob.pe/handle/20.500.12542/288) | 0,1° | Desde 1981 | Acceso público; la ficha no trae enlace de descarga. Existe también PISCOt de temperatura (no abrí su página) |
| Salidas WRF de 1 km (IGP) | [Saavedra et al. 2026, Atmospheric Research](https://repositorio.igp.gob.pe/items/458c9e91-e853-47fc-881a-fa7b008881d3) | 1 km (comparado con 5 km) | DJF 2001–2006, escenario de deforestación del 40 % | Es un antecedente peruano, no un dataset listo para usar. No encontré salidas publicadas y no es viable correr WRF en Colab gratuito |

## 5. Catálogos para seguir buscando

- [satellite-image-deep-learning/datasets](https://github.com/satellite-image-deep-learning/datasets): catálogo general de datasets de imágenes satelitales (de aquí salen varias entradas de arriba).
- [OpenForest](https://github.com/RolnickLab/OpenForest): catálogo de datasets forestales abiertos. La lista está en `OpenForest.csv`; no pude leerla.
- [awesome-forests](https://github.com/blutjens/awesome-forests): lista curada de datasets forestales.
- [TorchGeo](https://torchgeo.readthedocs.io/): cargadores de EuroSAT, BigEarthNet y SSL4EO-S12 (según la documentación que apareció en la búsqueda).

## 6. Modelo base (Meta)

- [DINOv3](https://ai.meta.com/dinov3/) y [blog con WRI](https://ai.meta.com/blog/world-resources-institute-dinov3/): WRI reporta que el error de altura de dosel bajó de 4,1 a 1,2 m en una región de Kenia. No confirmé que los pesos de la variante satelital estén publicados ni su licencia.
- [DINOv2](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/) y [arXiv 2304.07213](https://arxiv.org/abs/2304.07213) (mapa de altura de dosel).
- La página ai.meta.com/datasets no lista un dataset de deforestación.

## 7. Trabajos previos relevantes (réplica y árbol de problemas, 2022–2026)

IEEE (verificados abriendo la página):
- Soto Vega et al., *IEEE JSTARS* 16:10264–10278, 2023: [adversarial domain adaptation con supervisión débil para deforestación](https://doaj.org/article/e5d40fc1c31047f5bd2810759ee1d29a). Landsat-8, Amazonía y Cerrado.
- Castrillón-Candás et al., *IEEE TGRS* vol. 64, 2026: [deFOREST](https://arxiv.org/abs/2510.14092) (versión de autor aceptada). Sentinel-2 + Sentinel-1 en Jacundá (Brasil). Código en [GitHub](https://github.com/jcandas/deFOREST) (MATLAB).
- Sadel et al., *IEEE GRSL* 22, 2025: [Monitoring Forest Changes With Foundation Models and Sentinel-2 Time Series](https://research.polsl.pl/en/publications/monitoring-forest-changes-with-foundation-models-and-sentinel-2-t/). La página no nombra el modelo ni la región.

No IEEE (indexación en Scopus **no verificada**; confírmala en scopus.com/sources):
- McCallum et al. 2023, *Remote Sensing* 15(21):5204: [Crowd-Driven Deep Learning Tracks Amazon Deforestation](https://doi.org/10.3390/rs15215204). Entrena en Brasil y valida en Perú (84 %); ResNet18 fue el mejor modelo.
- Lee & Choi 2023, *Remote Sensing* 15(21):5167: [estrategia multivista con MultiEarth](https://www.mdpi.com/2072-4292/15/21/5167). Attention U-Net fue la mejor variante.
- Moura et al. 2025, *Forests* 16(5):742: [adaptación multi-dominio con incertidumbre](https://www.mdpi.com/1999-4907/16/5/742). Landsat-8, estados brasileños. Código: [GitHub](https://github.com/LPG-Uerj/FC-DANN_DA_For_CD_MultiTarget_TF2).
- Ortega et al. 2026, *PFG*: [domain adaptation con Sentinel-2](https://link.springer.com/article/10.1007/s41064-026-00413-x). Cuatro sitios brasileños, etiquetas PRODES a 10 m.
- Equipos de MultiEarth: [Team FOREVER](https://arxiv.org/abs/2306.11762) (F1 0,871, IoU 0,792 en una zona pequeña) y [SpaceVision4Amazon](https://arxiv.org/abs/2307.04715).
- Antecedente de simulación: Saavedra et al. 2026 (WRF, ver sección 4).

Observación: en los trabajos de adaptación de dominio que abrí, las regiones son brasileñas y ninguno menciona Perú, salvo McCallum, que usa una CNN supervisada. Esto no es una revisión exhaustiva; redáctalo como "hasta donde pudimos verificar".

## 8. Para decidir después

- ¿Cómo conseguir las muestras, dado el bloqueo a Zenodo? Opciones: descarga manual del usuario, un cuaderno de Colab, o seguir solo con documentación.
- ¿Cómo separar el efecto de región del efecto de etiquetado (Planet manual vs. Hansen/MapBiomas) y de estacionalidad (MultiEarth sin lluvias)?
- ¿Qué modelo Meta usar (DINOv2 o DINOv3), según los pesos disponibles?
- Aprobación del profesor (tema fuera del banco) y registro del póster antes del 16 de octubre.
