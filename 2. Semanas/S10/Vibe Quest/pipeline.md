# pipeline.md · Vibe Quest S10 (pregunta 3)

## Objetivo
Responder la pregunta 3 de S10 (SAE, monosemanticidad y steering) con un cuaderno interactivo para exponer en 5 minutos. Regla del curso: la IA es apoyo; debo poder explicar todo.

## Historial
1. Revisé S9, S10 y el resto de semanas (S11 a S15 vacías) y elegí la pregunta 3.
2. Fuentes: lectura asistida y notas propias (`fuentes/`), con estado de lectura por fuente.
3. Diseño del toy model: 12 rasgos, 4 neuronas, SAE de 24 latentes con L1; exploración de α (valores iniciales 0.003 y 0.03 recuperaban mal; se fijó α ∈ {0, 0.1, 0.3, 1.0}).
4. Ejecución en Colab (T4, redes en CPU): entrenamiento de 3 modelos y 12 SAE (141 s), robustez de 5 semillas (222 s totales del bloque), steering. `assert`s superados; la tabla se reprodujo idéntica en dos ejecuciones.
5. Exporté CSV; las figuras `fig1` y `fig2` se generan de esos CSV.
6. Redacté el cuaderno final, guion, interpretación y README.
7. Reestructuré el cuaderno al formato del Vibe Quest S9: 61 celdas (28 de código), con una celda de texto antes de cada celda de código y secciones `N · Nombre` (subsecciones `N · a · …`). Separé: librerías (1), parámetros y carpetas (2), datos (3), clases del modelo (4) y del SAE (5), funciones de entrenamiento (6), funciones de métricas y de dirección (7), invocación del entrenamiento (8), resultados (9), robustez (10), panel A y B con la función separada de su invocación (11, 12), steering por rasgo (13), y 14 a 17 hallazgos, ética, límites y referencias. Cada tabla, gráfico, impresión o guardado de archivo está en su propia celda.
8. Descargué los PDF de arXiv con el navegador integrado y verifiqué la primera página de cada uno (7 de 7).
9. Ejecución del código concatenado de las 23 celdas en Colab (ver sección de validación).

## Parámetros
N_RASGOS=12, N_DIM=4, K_DICT=24, ESPARSIDADES=[0.5, 0.9, 0.97], ALPHAS=[0, 0.1, 0.3, 1.0], PASOS_MODELO=4000, PASOS_SAE=5000, LOTE=2048, Adam (lr 0.005 modelo, 0.002 SAE), semilla 0.

## Resultados clave
- Representados: 8 / 11 / 12 de 12 (esparsidad 0.5 / 0.9 / 0.97).
- α = 0.3: recupera 1.00 (semilla 0) en 0.9 y 0.97; pureza de latentes 0.85 y 0.89 frente a 0.45 y 0.42 de neuronas.
- 5 semillas (0.97, α = 0.3): recupera 0.80 ± 0.14; selectividad SAE 0.81 ± 0.10 frente a 0.23 ± 0.03 neurona.
- Esparsidad 0.5: el SAE no mejora (máx. 0.375).

## Fuentes (ver `fuentes/README.md`)
Elhage 2022; Cunningham 2023; Bricken 2023; Templeton 2024; Arditi 2024; AxBench 2025; O'Brien 2024; Li y Kasneci 2026; Jørgensen y Hansen 2026; Clase 10.

## Limitaciones
Mundo sintético; hiperparámetros afinados en el mismo mundo; métricas propias; no se probó la variante "sumar vector"; fuentes de 2026 leídas solo en abstract e introducción; páginas de Anthropic leídas parcialmente (≈ 72 % y ≈ 78 %).

## Pendientes
- Leer los PDF completos (al menos Elhage y Templeton) antes de exponer. Bricken y Templeton no tienen PDF: URL en `fuentes/README.md`.
- Abrir el cuaderno en Colab y probar los dos paneles en vivo; ensayar con cronómetro.
- Opcional: ejecutar el `.py` completo de una vez para generar `panelA/panelB`.

## Regla de mantenimiento
Actualizar este archivo cada vez que se modifique código, datos, cuaderno o fuentes. No hay versiones v0X.
