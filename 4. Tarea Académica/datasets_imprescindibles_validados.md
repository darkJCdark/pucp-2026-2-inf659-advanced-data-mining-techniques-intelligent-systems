# Datasets imprescindibles: validación y uso en el proyecto

Proyecto: detección de deforestación amazónica con Sentinel-2, entrenada en Brasil y probada en Perú (zero-shot y ajuste fino corto). Curso INF659, WAIMLAp 2026. Validación hecha el 13 de octubre de 2026.

**Alcance de la validación.** Abrí la ficha oficial, el catálogo o el paper de cada dataset. **No descargué ningún archivo**: Zenodo está bloqueado por la lista de dominios permitidos, así que no pude abrir los datos. "Existe" significa que la ficha oficial existe y está vigente; "utilidad" es una evaluación mía sobre esa ficha. Lo que la ficha no dice figura como **no confirmado**.

## Resumen

| # | Dataset | ¿Existe? | Rol | Utilidad | Riesgo principal |
|---|---|---|---|---|---|
| 1 | MultiEarth 2023 | Sí (sitio del reto y paper) | Entrenamiento en Brasil | Alta, si se logra el acceso | El enlace de descarga publicado caducó el 12/06/2023 |
| 2 | Hansen Global Forest Change v1.13 | Sí (catálogo de Earth Engine) | Etiquetas de Perú (y de Brasil, en el plan B) | Alta | Etiqueta débil de 30 m; mide pérdida de cobertura arbórea, no solo deforestación |
| 3 | MapBiomas Alerta Perú | Sí (plataforma y página de método) | Etiquetas validadas de Perú | Media | No confirmé cómo se descarga ni la licencia |
| 4 | Sentinel-2 L2A (Earth Engine) | Sí (catálogo de Earth Engine) | Imagen de Perú y, en el plan B, de Brasil | Imprescindible | Cobertura L2A 2017–2018 no global |
| 5 | ERA5-Land | Sí (CDS y paper) | Covariable climática | Media-baja | Resolución de ~9 km frente a píxeles de 10 m |
| 5b | PISCO precipitación | Sí (paper y sitio) | Control de calidad en Perú | Baja para entrenamiento | Solo cubre Perú; licencia no indicada |

---

## 1. MultiEarth 2023

**Validación.**
- Existe: sitio del reto en [sites.google.com/view/rainforest-challenge/multiearth-2023](https://sites.google.com/view/rainforest-challenge/multiearth-2023). Contiene datos de entrenamiento de deforestación, Sentinel-1, Sentinel-2 (bandas 1–12) y Landsat 5 y 8, en NetCDF o ZIP con TIFF.
- Región: una zona de Pará, Brasil, de ~9.500 km² ([Lee & Choi 2023](https://www.mdpi.com/2072-4292/15/21/5167)). No incluye Perú.
- Etiquetas: máscaras derivadas de mosaicos Planet (3,7 m), 2016–2021, excluyendo la temporada de lluvias (octubre a abril) según Lee & Choi. Sentinel-2 cubre 2018–2021.
- **Acceso: no confirmado.** El comando de descarga con `azcopy` del sitio lleva una firma de acceso con vencimiento al 12 de junio de 2023 (así lo reportan el sitio y el repo [h2oai/cvpr-multiearth-deforestation-segmentation](https://github.com/h2oai/cvpr-multiearth-deforestation-segmentation), archivado desde el 2 de junio de 2026). Los plazos del reto también vencieron. Los enlaces directos ZIP/NetCDF podrían funcionar o no.
- Licencia: la página remite a una "Data License Table" externa. No confirmada. El paper dice que es de uso libre para desarrollo, investigación y educación.
- Tamaño: no indicado en la página.
- Contacto de los organizadores que figura en el sitio: multiearth2023@gmail.com.

**Utilidad.** Es el único conjunto que encontré con imagen Sentinel-2 y máscara de deforestación ya alineadas. Hay trabajo previo replicable: el repo de h2oai (licencia MIT) reporta F1 0,871 e IoU 0,792 en el área de prueba. Pero cada experimento de ese repo tarda 2–3 horas con 3 GPU Quadro RTX 6000, lejos de Colab gratuito. Con un backbone congelado y un subconjunto de parches sí es viable.

**Cómo se usaría.**
1. Extraer parches Sentinel-2 de 256×256 con su máscara.
2. Extraer características con DINOv2/v3 congelado y entrenar solo una cabeza liviana.
3. Reportar F1 e IoU en una partición de Pará como línea base y comparar con los números del repo de h2oai.
4. Congelar ese modelo y evaluarlo en Perú (zero-shot).

**Paper asociado.**
- Cha et al., *MultiEarth 2023 – Multimodal Learning for Earth and Environment Workshop and Challenge*, 2023. Es un workshop de CVPR. arXiv:2306.04738 (DOI de arXiv: 10.48550/arXiv.2306.04738). No encontré un DOI de actas distinto. [Enlace](https://arxiv.org/abs/2306.04738).
- Trabajo sobre el mismo dataset: Lee & Choi 2023, *Remote Sensing* 15(21):5167, DOI [10.3390/rs15215167](https://doi.org/10.3390/rs15215167).

**Enlace al dataset.** [Sitio del reto](https://sites.google.com/view/rainforest-challenge/multiearth-2023).

**Si no hay acceso (plan B).** Construir el conjunto de entrenamiento propio con Sentinel-2 y Hansen sobre Pará (ver secciones 2 y 4). Tiene una ventaja: las etiquetas de Brasil y Perú vendrían del mismo producto, y así el desempeño en Perú no se mezcla con un cambio de fuente de etiquetas.

---

## 2. Hansen Global Forest Change (v1.13)

**Validación.**
- Existe y está vigente: [catálogo de Earth Engine, v1.13](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2025_v1_13). Cubre 2000–2025, resolución 30,92 m. La v1.12 figura como obsoleta.
- ID: `UMD/hansen/global_forest_change_2025_v1_13`.
- Bandas: `treecover2000`, `loss`, `lossyear` (1–25 = 2001–2025), `gain` (solo 2000–2012), compuestos de referencia y `datamask`.
- Licencia: CC-BY-4.0.
- Acceso: por Earth Engine. La ficha no indica requisitos de cuenta; verifica las condiciones de uso de Earth Engine para tu caso.

**Utilidad.** Cubre Perú y Brasil con el mismo método, lo que permite etiquetar ambos dominios por igual. Limitación: mide pérdida de cobertura arbórea a 30 m, no deforestación estricta, y debe cruzarse con un umbral de `treecover2000` para definir "bosque". Confirma este punto leyendo el paper antes de fijar la definición de la etiqueta. Las etiquetas serán débiles frente a los 10 m de Sentinel-2.

**Cómo se usaría.**
1. Elegir un umbral de `treecover2000` para definir bosque.
2. Marcar como "deforestado" los píxeles con `loss = 1` y `lossyear` en el periodo de las imágenes Sentinel-2.
3. Re-muestrear a la grilla de 10 m de Sentinel-2 y generar parches con su máscara.
4. En Perú, estas son las etiquetas de la prueba zero-shot; en el plan B, también del entrenamiento en Brasil.
5. Analizar el ruido de la etiqueta (omisiones, fecha del cambio a nivel anual) como parte del análisis de fallos.

**Paper asociado.** Hansen et al. 2013, *High-Resolution Global Maps of 21st-Century Forest Cover Change*, *Science* 342:850–853, DOI [10.1126/science.1244693](https://doi.org/10.1126/science.1244693) (la cita figura así en el catálogo).

**Enlace al dataset.** [Earth Engine v1.13](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2025_v1_13).

---

## 3. MapBiomas Alerta Perú

**Validación.**
- Existe: [plataforma](https://plataforma.peru.alerta.mapbiomas.org/mapa) y [página de método](https://peru.alerta.mapbiomas.org/en/?p=92).
- Método según esa página: las alertas tempranas vienen del sistema oficial de Perú (Landsat, 30 m, bioma amazónico). Se descartan las menores de 1 ha, se inspeccionan visualmente con mosaicos Planet (3,7 m) y Sentinel-2 de apoyo, y las confirmadas se refinan (Random Forest o manual). Se actualiza semanalmente y pasan 30–60 días entre detección y publicación.
- **Descarga, formato, licencia y periodo: no confirmados.** Las páginas que abrí no los indican. La plataforma es una aplicación web que no pude explorar. La página de [descargas de MapBiomas Perú](https://peru.mapbiomas.org/en/descargas) (CC BY 4.0) cubre otros productos (cobertura por colección, mosaicos Landsat), no confirmé que incluya estas alertas.

**Utilidad.** Sus polígonos están validados contra imagen de 3,7 m, así que podrían ser mejores que Hansen como verdad de campo en Perú. Si no se pueden descargar, sirve solo como consulta visual para auditar casos concretos.

**Cómo se usaría.**
1. Consultar o descargar polígonos de alerta validados para una zona de Perú.
2. Compararlos contra las máscaras de Hansen sobre las mismas escenas: concordancia y discrepancias.
3. Usarlos como etiqueta "más limpia" en un subconjunto pequeño de evaluación.

**Paper asociado.** No encontré un paper revisado por pares específico de MapBiomas Alerta Perú. La referencia es la [página de método](https://peru.alerta.mapbiomas.org/en/?p=92), sin DOI. Si lo necesitas, pregunta a MapBiomas cómo citarlo.

**Enlace al dataset.** [Plataforma MapBiomas Alerta Perú](https://plataforma.peru.alerta.mapbiomas.org/mapa).

---

## 4. Sentinel-2 L2A (imagen)

**Validación.**
- Existe y está vigente: [Earth Engine `COPERNICUS/S2_SR_HARMONIZED`](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED), disponible del 28 de marzo de 2017 al 12 de octubre de 2026 (fecha de la ficha), revisita de 5 días.
- 12 bandas de reflectancia superficial: 10 m (B2, B3, B4, B8), 20 m (B5–B7, B8A, B11, B12, SCL) y 60 m (B1, B9). Incluye `SCL` (máscara de nubes y clases).
- Advertencia de la ficha: la cobertura L2A de 2017–2018 no es global en Earth Engine. La colección armonizada corrige el desplazamiento de 1000 introducido desde el 25 de enero de 2022.
- Licencia: términos de datos Copernicus Sentinel. La ficha no ofrece formato de cita.
- La ficha no menciona Perú; la extensión es de –56° a 83° de latitud.

**Utilidad.** Imprescindible. Las etiquetas de Hansen y MapBiomas no traen imagen, y para Perú no hay otro dataset listo con imagen y etiqueta que yo haya verificado. También permite construir el entrenamiento en Brasil (plan B).

**Cómo se usaría.**
1. Elegir un compuesto sin nubes por fecha o ventana (con `SCL`), sobre las zonas de Perú y de Pará.
2. Exportar parches pequeños alineados con las etiquetas de Hansen.
3. Probar variantes de entrada: solo RGB frente a más bandas (ablación).
4. Ojo con las nubes: en la Amazonía son un confusor importante y conviene reportar el desempeño por nivel de nubosidad.

**Paper asociado.** Drusch et al. 2012, *SENTINEL-2: ESA's Optical High-Resolution Mission for GMES Operational Services*, *Remote Sensing of Environment* 120:25–36 (título, volumen y páginas confirmados en la [página de referencias de ESA](https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/references)). DOI: [10.1016/j.rse.2011.11.026](https://doi.org/10.1016/j.rse.2011.11.026). El DOI redirige a la ficha de Elsevier, pero no pude leer esa ficha para confirmar que coincide con el título.

**Enlace al dataset.** [Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED).

---

## 5. ERA5-Land (clima)

**Validación.**
- Existe: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land). Resolución 0,1° (~9 km nativos), horario, 1950–presente, licencia CC-BY, DOI del dataset 10.24381/cds.e2161bac.
- Descarga con la API `cdsapi`. La ficha no aclara si se necesita una cuenta del CDS: confírmalo al registrarte.

**Utilidad: media-baja, con una advertencia.** Un píxel de ERA5-Land equivale a unos 900×900 píxeles de Sentinel-2. Un parche de 256×256 a 10 m (~2,5 km de lado) cabe dentro de una sola celda. Por eso, como "canal de entrada" aporta un valor constante por parche, no detalle espacial. Es contexto (precipitación, humedad del suelo, temperatura del mes previo), no fusión de imágenes. Hay un riesgo adicional: como el clima difiere entre Pará y Perú, el modelo podría usarlo como atajo para "reconocer la región".

**Cómo se usaría.**
1. Extraer, para el centro de cada parche y la fecha de su imagen, unas pocas variables (precipitación, humedad del suelo, temperatura 2 m) de ERA5-Land.
2. Concatenarlas como vector junto a las características de imagen, con una cabeza adicional.
3. Ablación: con y sin clima, tanto en Brasil como en Perú, para ver si ayuda o solo permite el atajo.

**Paper asociado.** Muñoz-Sabater et al. 2021, *ERA5-Land: a state-of-the-art global reanalysis dataset for land applications*, *Earth System Science Data* 13(9):4349–4383, DOI [10.5194/essd-13-4349-2021](https://doi.org/10.5194/essd-13-4349-2021).

**Enlace al dataset.** [CDS, ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) (DOI: [10.24381/cds.e2161bac](https://doi.org/10.24381/cds.e2161bac)).

### 5b. PISCO precipitación (SENAMHI): solo control de calidad

- Existe: [piscoprec.github.io](https://piscoprec.github.io/). PISCOp V2.1, 0,1°, desde 1981; la web indica descarga por Google Earth Engine y por FTP de SENAMHI (sin enlaces directos ni formato). Licencia no indicada. El paper dice que el dataset es público.
- Utilidad baja para el entrenamiento: **solo cubre Perú**, así que no puede ser una entrada común a ambos dominios. Se puede usar para comprobar en Perú que ERA5-Land reproduce bien la precipitación local.
- Paper: Aybar et al. 2020, *Construction of a high-resolution gridded rainfall dataset for Peru from 1981 to the present day*, *Hydrological Sciences Journal* 65(5):770–785, DOI [10.1080/02626667.2019.1649411](https://doi.org/10.1080/02626667.2019.1649411).
- Enlace al dataset: [piscoprec.github.io](https://piscoprec.github.io/).

---

## Orden recomendado de adquisición

1. **Sentinel-2 y Hansen en Earth Engine.** No dependen de terceros y permiten empezar ya, incluyendo el plan B de entrenamiento en Pará.
2. **MultiEarth.** Escribir ya a multiearth2023@gmail.com para confirmar acceso y licencia, y probar los enlaces directos.
3. **MapBiomas Alerta Perú.** Explorar la plataforma y consultar cómo descargar los polígonos y con qué licencia.
4. **ERA5-Land.** Abrir cuenta del CDS y descargar solo las variables y fechas necesarias.

## Pendiente de confirmar

- Si MultiEarth se puede descargar hoy y bajo qué licencia.
- Cómo se descargan las alertas de MapBiomas Alerta Perú y su licencia.
- Si Earth Engine admite el uso previsto (cuenta y condiciones).
- La definición exacta de "pérdida" en Hansen (leer el paper).
- Aprobación del profesor para un tema fuera del banco, y registro del póster antes del 16 de octubre.
