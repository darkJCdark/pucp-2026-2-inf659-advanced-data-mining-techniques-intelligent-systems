# resultados · google colab

Ejecución completa de `vibe_quest_s13_transformer_sin_pe.ipynb` en Google Colab con GPU T4 (torch 2.14.1+cu130).

| Archivo | Contenido |
| --- | --- |
| `e1_consigna_torch.csv`, `e2_resumen_torch.csv`, `e3_permutaciones_torch.csv`, `chequeos_hipotesis_torch.csv` | Parte A (frases de la consigna, barrido, permutaciones y chequeos) |
| `B_resumen_datos.csv` | Tamaño de cada tarea, vocabulario y balance de etiquetas |
| `B_invarianza_previa.csv` | Diferencia al permutar las palabras antes de entrenar (sección 3.9) |
| `B_resultados_por_semilla.csv` | Los 24 entrenamientos: exactitud en prueba y quitando el PE |
| `B_ablacion_pe_resumen.csv` | Exactitud media y desviación al escalar el PE de 0 a 1 en evaluación |
| `B_consigna_modelos_entrenados_resumen.csv` | La consigna con los modelos entrenados (sección 3.18), media de 3 semillas |
| `B_chequeos_hipotesis.csv` | Los 7 chequeos de la Parte B (todos verdaderos) |

Las curvas de validación por época (`B_curvas_validacion.csv`) y las tablas completas las guarda la sección 7 del notebook dentro de `resultados_s13.zip`.
