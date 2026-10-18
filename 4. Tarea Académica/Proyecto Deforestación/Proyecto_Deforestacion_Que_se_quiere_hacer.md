# Proyecto de deforestación (TA final INF659): qué se quiere hacer y en qué estado está

Estado al 18 de octubre de 2026. Equipo: Christian Carhuancho, Jorge Chamorro y Jorge Fatama. Evento: WAIMLAp 2026 (18–20 de noviembre, Lima).

## 1. En una frase

Se quiere medir si las características de un modelo visual preentrenado y congelado (DINOv2) detectan pérdida de bosque en la Amazonía peruana cuando el clasificador se entrenó solo en Pará (Brasil), y compararlo con una CNN supervisada (EfficientNet-B0). El aporte es el análisis: cuánto cae el desempeño al cambiar de país, en qué capa del modelo aparece la señal y por qué falla.

## 2. Qué revisé y qué encontré

Revisé la carpeta del curso (`4. Tarea Académica/Entregables-…/Entregables`), el historial de git, la memoria del proyecto y los documentos del Project. No pude ver el video `2026-10-15 09-08-04.mkv` (28 MB). Los archivos de la TA son:

| Archivo | De qué trata | Estado |
|---|---|---|
| `E1. Propuesta/Propuesta.docx` | Propuesta E1 sobre **deepfakes con DFDC** (EfficientNet-B0 vs. DINOv2 ViT-S/14) | Es el entregable E1, tema anterior |
| `E1_EfficientNetB0_DFDC.ipynb` y `E1_EfficientNetB0_DINOv2_DFDC.ipynb` | Cuadernos E1 ejecutados sobre DFDC (GPU RTX 3060) | Con resultados guardados; tema anterior |
| `E1. Propuesta/WAIMLAp_2026_Abstract.docx` | Abstract de deforestación, versión A: **DINOv3 congelado vs. U-Net**, entrenamiento solo con MultiEarth | Desfasado frente a la versión B |
| `[Cursor]Abstract_v1.docx` | Abstract de deforestación, versión B: **DINOv2 ViT-S/14 congelado vs. EfficientNet-B0**, etiquetas Hansen, sitios en Pará y Madre de Dios | Es la versión más completa; declara "no experiment has been run" |
| `Chat con Cursor - 14.10.docx` | Conversación con Cursor que fija las reglas de ventanas, umbrales y etiqueta | Cierra los huecos de la versión B |
| `Datasets.docx` | El archivo de datasets validados que hice (versión mezclada con la lista general) | Su encabezado dice "usando Dino V3" |
| `WAIMLAp_2026_A0_Poster.pptx` y `_v1.pptx` | Póster A0 con el título de la versión A | `_v1` tiene textos de motivación y objetivos; resultados y metodología vacíos; el otro es la plantilla |

No encontré en la carpeta el archivo `Abstract_FrozenDINOv2_ForestLoss_Para_Peru.docx` que el chat dice que quedó actualizado con las reglas. Tampoco hay ningún cuaderno ni script de deforestación.

## 3. Qué se quiere hacer (versión B, la más reciente)

**Pregunta.** ¿Un sondeo lineal entrenado solo en Pará sobre características congeladas de DINOv2 sigue detectando pérdida de bosque en la Amazonía peruana?

**Datos.**
- Imagen: Sentinel-2 L2A a 10 m, compuesto de mediana de junio a septiembre, años 2022, 2023 y 2024, con la capa de clasificación de escena (SCL) para quitar nubes.
- Etiqueta: Hansen Global Forest Change v1.13 en ambos países, con la misma regla. Así, una caída en Perú no se debe a un cambio de fuente de etiquetas.
- MultiEarth queda descartado como etiqueta de entrenamiento: solo cubre Pará, usa otro sensor y sus enlaces públicos caducaron (coincide con lo que encontré al validar).

**Sitios (ventanas de 0,5° × 0,5°, ~55 km).**
- Pará: 6,5°S–7,0°S, 52,0°O–51,5°O, frente de São Félix do Xingu. Clareos grandes de frontera agropecuaria.
- Perú: 12,4°S–12,9°S, 70,1°O–69,6°O, corredor de la Interoceánica en Madre de Dios. Clareos más chicos y minería.
- Cada ventana puede desplazarse una sola vez, dentro del mismo departamento, si no reúne 200 parches positivos y 200 negativos. El desplazamiento se anota y no se decide mirando al modelo.
- El terreno se parte en bloques de 5 km y ningún parche cruza de bloque.

**Regla de etiqueta (igual en los dos países, fijada antes de medir).**
- Bosque: `treecover2000` ≥ 30 %. Los umbrales de 50 % y 80 % se reportan después solo como sensibilidad.
- Parche de 224×224 píxeles a 10 m (2,24 km, ~502 ha). Positivo: ≥ 6,25 ha de pérdida en el año del compuesto. Negativo: < 1 ha de pérdida y al menos la mitad del parche en bosque en 2000.
- La banda de 1 a 6,25 ha queda fuera de la métrica principal y se reporta como estrés de clareo pequeño.
- La pérdida cuenta solo si `lossyear` es el año del compuesto menos 2000. Los años vecinos son el estrés de año ambiguo.

**Modelos.**
- DINOv2 ViT-S/14 congelado, con un sondeo lineal sobre el token de clase, repetido en cada uno de sus 12 bloques.
- EfficientNet-B0 entrenado con los mismos parches de Pará, con un esquema corto y fijo.
- Entrada principal RGB. Una ablación: un adaptador 1×1 que lleva bandas extra de Sentinel-2 a 3 canales, ajustado solo en Pará.

**Cuatro protocolos, declarados antes de medir.**
1. Bloques retenidos dentro de Pará.
2. Bloques de Perú que el sondeo nunca vio (zero-shot).
3. Actualización corta del sondeo y de los últimos bloques con un subconjunto pequeño de Perú, sorteado de antemano, con los bloques de prueba de Perú retenidos.
4. Cortes de estrés: nube alta, clareo pequeño y año ambiguo.

**Métricas.** ROC AUC y F1 por parche. El umbral de F1 se elige en la validación de Pará y se congela. Se planean varias semillas; todavía no se corrió ninguna.

**Hipótesis fijadas antes de medir.** En Pará, la CNN igualará o superará al sondeo congelado. En Perú, el sondeo caerá menos. Una señal de pérdida aparecerá antes del último bloque.

**Ética (ya incluida en el borrador).** Una falsa alarma puede marcar a una comunidad que no taló; una omisión deja pasar tala ilegal. No se publicarán sitios a resolución de vivienda, y una etiqueta Hansen es pérdida de cobertura arbórea, no una conclusión legal.

## 4. Cómo encaja con lo que exige el curso

| Requisito de la TA | Cómo lo cubre la versión B |
|---|---|
| Fase A: disección (parámetros y FLOPs por bloque, latencia, memoria) | Cuentas por bloque y barrido de sondeos por los 12 bloques de DINOv2; B0 con ~4 M de parámetros vs. ~22 M de DINOv2 |
| Fase B: intervención con hipótesis previa y varias semillas | Sondeo por bloque, adaptador de bandas, actualización corta; falta correr semillas |
| Fase C: estrés fuera de dominio y fallos honestos | Pará → Perú, tres cortes de estrés, resultados negativos reportados |
| Ética y limitaciones | Párrafo incluido en el borrador |
| Cuaderno de decisiones con uso de IA | Ya se usa en los cuadernos E1 (con Cursor); hay que continuarlo en el proyecto nuevo |
| Tema fuera del banco | Requiere aprobación del profesor; no encontré constancia de que se haya pedido |

**Lo reutilizable de E1.** Los cuadernos de DFDC ya tienen el teardown de EfficientNet-B0 y de DINOv2 ViT-S/14, el sondeo por bloque, el bootstrap y las pruebas de degradación. Según sus salidas guardadas: B0 tiene 4.010.110 parámetros y 0,3845 G MACs; DINOv2 tiene 22.056.576 parámetros y 5,51 G MACs a 224 px. Latencia con lote 16: B0 13,2 ms y DINOv2 41,5 ms. En DFDC el AUC de prueba de B0 fue 0,9970 y el de DINOv2 por bloque llegó a su máximo en el bloque 11 (0,9928). Solo cambian los datos.

## 5. Contradicciones que hay que cerrar (lo más urgente)

1. **Modelo.** La versión A y el póster dicen DINOv3; la versión B y el chat dicen DINOv2 ViT-S/14. No verifiqué que los pesos de DINOv3 satelital estén disponibles. DINOv2 ViT-S/14 es abierto y cabe en Colab gratuito.
2. **Línea base.** U-Net supervisada (A) frente a EfficientNet-B0 (B).
3. **Tarea.** Segmentación con máscaras y métricas F1/IoU (A) frente a clasificación por parche con sondeo lineal y ROC AUC (B). Con B ya no se puede comparar F1/IoU contra los resultados publicados de MultiEarth.
4. **Datos.** "MultiEarth es la fuente primaria y única de entrenamiento" (A) frente a Sentinel-2 + Hansen (B).
5. **Lenguaje.** La versión A habla de zero-shot, few-shot y ajuste corto a la vez; la B los separa en protocolos 2 y 3. Usa los términos de la B.
6. **Archivos desfasados.** El título del póster y el abstract A siguen siendo de la versión A. `Datasets.docx` dice "Dino V3". El abstract final con las reglas no está en la carpeta.
7. **ERA5/PISCO y fusión multifuente.** En una conversación anterior decidiste mantenerlos como canales de entrada. Ninguna versión de los abstracts los incluye. Hay que decidir si quedan fuera del protocolo principal (mi recomendación, por su resolución de ~9 km frente a parches de 2,24 km) o si se agregan como ablación.

## 6. Verificación de las reglas del chat con Cursor

| Afirmación | Resultado |
|---|---|
| El Global Forest Review usa por defecto copa > 30 % en 2000 | Confirmado: "use a baseline of tree cover with greater than 30 percent canopy density in the year 2000" ([GFR](https://gfr.wri.org/annual-tree-cover-loss-data-explained)) |
| PRODES mapea desde 6,25 ha | Confirmado en [arXiv 2211.09806](https://arxiv.org/abs/2211.09806): "The minimum area considered for mapping deforestation is 6.25 ha" |
| Gasparini et al.: 80 % se acerca más al bosque de PRODES | Confirmado: Gasparini et al. 2019, *Sensors* 19(22):5020, DOI [10.3390/s19225020](https://doi.org/10.3390/s19225020). Matiz: el estudio es de Mato Grosso, no de la Amazonía en general |
| McCallum: 84 % de concordancia sobre 100 imágenes Sentinel-2 de 2019 en Perú | Confirmado el dato de las 100 imágenes y del 84 %; el texto principal no explica cómo se calculó (remite a la Tabla S5). Compara contra el mapa global de pérdida anual (lossyear de Hansen/GLAD) |
| Las ventanas corresponden a São Félix do Xingu y a la Interoceánica en Madre de Dios | No lo verifiqué; revisa las ventanas en un mapa |
| Habrá ≥ 200 positivos y ≥ 200 negativos por ventana | No verificable hasta consultar Earth Engine (ver riesgo b) |

## 7. Riesgos del diseño (mi lectura)

a) **Perú queda representado por una sola ventana de 55 km** y con otro motor de cambio (minería y clareos chicos). Cuando el sondeo falle en Madre de Dios no sabrás si es "país", "tipo de clareo" o "tipo de motor". Conviene nombrarlo así en el título y las conclusiones, o sumar una segunda ventana peruana.
b) **Positivos en Perú.** Una ventana de 55 km da unos 600 parches de 2,24 km (cálculo mío); con 3 años, unos 1.800. Para llegar a 200 positivos con ≥ 6,25 ha de pérdida por parche haría falta que ~11 % de los parches califique. En clareos pequeños puede no alcanzar. Por eso existe la regla de desplazar una vez.
c) **Etiqueta débil.** Hansen es anual y de 30 m sobre imagen de 10 m. El propio borrador lo asume; hay que analizarlo con casos concretos, no solo declararlo.
d) **Umbral de 30 %.** Gasparini sugiere que 30 % también marca vegetación abierta. El plan lo contempla con la sensibilidad a 50 % y 80 %, bien.
e) **Alcance.** Una semilla de DFDC no valida nada para deforestación. Aquí hay que correr varias semillas.
f) **Calendario.** Faltan 29 días para la entrega del 16 de noviembre. El registro del póster vencía el 16 de octubre y no sé si se hizo.

## 8. Qué decidir esta semana

1. Elegir una versión (recomiendo la B) y marcar la A como reemplazada.
2. Confirmar con el profesor el cambio de tema y si E1 (DFDC) cuenta como está.
3. Confirmar el registro del póster (venció el 16 de octubre).
4. Decidir qué pasa con ERA5/PISCO y la fusión multifuente.
5. Verificar en Earth Engine las ventanas y el conteo de parches antes de mirar cualquier puntaje.
6. Actualizar póster, abstract y `Datasets.docx` con el modelo y la línea base elegidos.

## 9. Fuentes de datos y papers (de la validación del 13 de octubre)

- Hansen GFC v1.13: [catálogo Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2025_v1_13); Hansen et al. 2013, *Science* 342:850–853, DOI [10.1126/science.1244693](https://doi.org/10.1126/science.1244693).
- Sentinel-2 L2A: [Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED); Drusch et al. 2012, *Remote Sens. Environ.* 120:25–36, DOI [10.1016/j.rse.2011.11.026](https://doi.org/10.1016/j.rse.2011.11.026).
- McCallum et al. 2023, *Remote Sensing* 15(21):5204, DOI [10.3390/rs15215204](https://doi.org/10.3390/rs15215204).
- Soto Vega et al. 2023, *IEEE JSTARS* 16:10264–10278 ([ficha](https://doaj.org/article/e5d40fc1c31047f5bd2810759ee1d29a)).
- MultiEarth 2023: [sitio del reto](https://sites.google.com/view/rainforest-challenge/multiearth-2023) y [paper](https://arxiv.org/abs/2306.04738) (acceso no confirmado).
