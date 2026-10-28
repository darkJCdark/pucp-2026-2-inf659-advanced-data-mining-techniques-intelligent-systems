# pipeline.md · Vibe Quest S11 (pregunta 1: gradientes que se desvanecen y explotan)

## Objetivo
Responder: *¿cómo afectan los gradientes que se desvanecen al rendimiento de una RNN simple en una tarea con dependencias a largo plazo, y se puede demostrar con código cómo se desvanecen o explotan frente a un modelo normal?* Presentación de 5 minutos con sliders.

## Flujo (entrada → modelos → salida)

```
ENTRADA                          MODELOS (numpy, BPTT a mano)                 SALIDA
tarea_memoria.generar(T)         ┌ RNN simple  h_t = tanh(x_t Wx + h_{t-1} Wh + b)     tablas (datos/)
 t=0: one-hot de la clase    ──► │ RNN + recorte de gradiente (norma máx. 1.0)     ──►  aritmetica_lamina.csv
      + indicador                │ LSTM (i, f, o, g; sesgo de olvido 2.0)               normas_gradiente_inicio.csv
 t>0: ruido N(0, 0.5²)           └ todos: logits = h_T Wy + by, entropía cruzada        resultados_entrenamiento.csv
 etiqueta = clase de t=0                                                              resumen_entrenamiento.csv
                                 Adam lr 0.005, lote 64, 3000 pasos, 3 semillas       ajuste_hiperparametros.csv
                                                                                      figuras (capturas/)
                                 medidas en cada paso:                                fig1_gradiente_por_distancia.png
                                 ‖dL/dh_t‖ por paso temporal t                        fig2_exactitud_vs_longitud.png
                                 norma global del gradiente
```

## Historial de pasos
1. Revisé las láminas y la transcripción de la Clase 11 (`2. Semanas/S11`): BPTT, w^(T−1), recorte, LSTM, GRU. Las 4 preguntas del Vibe Quest están en la lámina final.
2. Elegí la pregunta 1 (decisión acordada contigo): es la única con fuentes IEEE 2022-2026 conceptuales y se ejecuta en numpy sin torch.
3. Fuentes: restricción acordada, solo Scopus o IEEE Xplore 2022-2026; las clásicas solo "citado en". Ver `fuentes/README.md`. Los 6 PDF están en `fuentes/`.
4. Código: `modelos_numpy.py` (RNN, LSTM, Adam, recorte), `tarea_memoria.py`, `verificar_gradientes.py` (BPTT contra diferencias finitas: error 9e-10 en RNN y 1.6e-7 en LSTM), `ajuste_hiperparametros.py`, `experimento_gradientes.py`, `descargar_fuentes.py`.
5. Ajuste con la semilla 100 (distinta de las de evaluación 0, 1, 2) a T = 30: lr ∈ {0.002, 0.005, 0.01, 0.02}; RNN con radio de Wh ∈ {1.0, 1.2}; LSTM con sesgo de olvido ∈ {1, 2}. Elegí lr 0.005, radio 1.0 y sesgo 2.0. La LSTM con sesgo 1.0 falló con lr 0.01 y 0.02; con 2.0 acertó en los tres lr.
6. Ejecución final: 54 entrenamientos (3 modelos × 6 longitudes × 3 semillas), 548 s en CPU.
7. Notebook `vibe_quest_s11_gradientes_rnn_lstm.ipynb` (PyTorch, 57 celdas: 31 de texto con `#` y 26 de código, cada código separado). Misma tarea y parámetros que NumPy. Incluye verificaciones con `assert` (recorrido manual = modelo, w^49 con autograd), 3 paneles con sliders (A gradiente sin entrenar, B curvas de aprendizaje, C entrenamiento en vivo), chequeos de hipótesis con True/False, comparación con la referencia NumPy, tabla afirmación ↔ fuente, interpretación y una celda final que guarda y comprime los resultados.
8. Validación en Google Colab (torch 2.11.0, CPU): el notebook corre completo sin errores y los `assert` pasan. Probé los sliders: Panel A con modelo, T (10 a 100), radio (0.5 a 2.5) y semilla, por clic y por código (6 combinaciones); Panel B con las 18 combinaciones de T y semilla; Panel C con el botón Entrenar (RNN y LSTM, con y sin recorte). Con T = 100 y radio 0.5 el gradiente de t = 1 se anula en float32 y el título lo indica.
9. Dos ejecuciones completas en Colab con las mismas semillas dieron cifras distintas (1 018 s y 803 s). Ejemplo: RNN con T = 80, media 0.64 la primera vez y 0.25 la segunda; RNN con T = 50, 0.64 y 0.92. La LSTM con la semilla 1 quedó en el azar en T = 50 y 80 en ambas. Por eso la conclusión "la RNN falla de forma consistente desde T ≥ 50 y la LSTM la supera" **no está respaldada con PyTorch**. La sección 15 del notebook la reemplaza por: el desvanecimiento se observa cuando la RNN falla (razón de gradientes de 1e-5 a 1e-30) y el entrenamiento depende de la semilla desde T = 20 en ambos modelos; con 3 semillas no se puede decir cuál modelo es mejor.
10. Salidas de Colab guardadas en `resultados/google colab/` (4 CSV de la segunda ejecución). Las dos figuras no se descargaron; la sección 16 del notebook las comprime en `resultados_s11.zip` y la descarga.
11. Guion de 5 minutos con tiempos, qué slider mover y preguntas posibles: `guion_5min.md`. Se presenta con la conclusión prudente (sección 15 del notebook).

## Parámetros
N_CLASES=4, entrada 5 canales, OCULTAS=32, LOTE=64, PASOS=3000, LR=0.005 (Adam), UMBRAL_CLIP=1.0, SEMILLAS=[0,1,2], T ∈ {5,10,20,30,50,80}, 2000 muestras de prueba (semilla 999).

## Resultados clave
- Lámina: 1.1^50 ≈ 117.39 y 0.9^50 ≈ 0.00515 (con exponente 49, como dice la fórmula w^(T−1): 106.72 y 0.00573).
- Sin entrenar, T = 50, ‖dL/dh₁‖/‖dL/dh₅₀‖: RNN radio 0.9 → 1.5e-7; radio 1.0 → 7.4e-6; radio 1.5 → 0.83; radio 2.0 → 364; LSTM → 0.53.
- Exactitud media (3 semillas), RNN / LSTM: T=5, 10 → 1.00 / 1.00; T=20 → 0.58 / 1.00; T=30 → 0.69 / 1.00; T=50 → 0.26 / 0.75; T=80 → 0.33 / 1.00. Azar = 0.25.
- La RNN solo alcanza ≥ 0.95 en 2 de 12 corridas con T ≥ 20; la LSTM en 17 de 18 en total (falla T=50, semilla 1: 0.256).
- Mediana de log10(‖dL/dh₁‖/‖dL/dh_T‖) durante el entrenamiento: RNN ≈ −4.7 (T=20) a −10.3 (T=80); LSTM entre −0.8 y +0.07.
- El recorte de gradiente no resuelve el desvanecimiento: con T=80 queda en 0.32 como la RNN sin recorte.

## Fuentes (ver `fuentes/README.md`)
Seis artículos IEEE (PDF en `fuentes/`): Tolić 2024, He 2022, Dou y Principe 2023, Ceni 2025, Rehmer y Kroll 2022, Shan 2023; leídas las partes citadas. Material del curso: Clase 11.

## Limitaciones
Tarea sintética diseñada por mí (no viene de las fuentes); 3 semillas (alta varianza: la LSTM falló una vez); hiperparámetros ajustados en el mismo tipo de tarea; un solo tamaño oculto (32); la GRU no se evalúa; la métrica de cociente de gradiente usa el promedio de normas por muestra.

## Regla de mantenimiento
Actualizar este archivo cada vez que se modifique código, datos, fuentes o cuaderno. No hay versiones v0X.
