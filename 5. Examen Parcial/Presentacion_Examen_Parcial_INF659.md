# Presentación — Examen Parcial INF659 (2026-2)

**Duración máxima:** 10 min · **Objetivo:** demostrar que entiendes cada decisión y cada resultado (los puntos se acreditan solo si lo explicas).
**Formato sugerido:** 11 diapositivas + mostrar 2 o 3 celdas del notebook ejecutado en vivo (marcadas con 🖥️).

| # | Diapositiva | Tiempo | Acumulado |
|---|---|---|---|
| 1 | Portada y contexto | 0:30 | 0:30 |
| 2 | P1 · Modelo base y parámetros | 0:40 | 1:10 |
| 3 | P1 · Predicción vs realidad | 0:35 | 1:45 |
| 4 | P2 · Tabla de ablación | 1:00 | 2:45 |
| 5 | P2 · Por qué cambian los resultados | 1:15 | 4:00 |
| 6 | P3 · Estrategia slice-wise | 0:45 | 4:45 |
| 7 | P3 · Feature extraction vs fine-tuning | 1:00 | 5:45 |
| 8 | P4 · Grad-CAM | 1:30 | 7:15 |
| 9 | P5 · Fiabilidad y BatchNorm | 1:00 | 8:15 |
| 10 | P5 · Métrica clínica y modo de fallo | 1:00 | 9:15 |
| 11 | Cierre: qué no funcionó y uso de IA | 0:30 | 9:45 |

Margen: 15 s.

---

## 1. Portada y contexto (0:30)

**En la diapositiva**
- Examen Parcial INF659 — Clasificación de TC 3D (COVID vs normal)
- Nombre y código
- Datos: 200 volúmenes (100/100), 128×128×64, HU [-1000, 400] → [0, 1]
- Split estratificado: 170 train / 30 test (test también usado como validación)
- `batch_size=2`, Adam + `ExponentialDecay`, **10 épocas** (reducción declarada)

**Qué decir**
- Por qué entrené 10 épocas: se cayó la sesión de Colab y lo apliqué igual a todas las variantes para que la comparación fuera justa.

---

## 2. P1 · Modelo base y parámetros (0:40)

**En la diapositiva**
- 3 × [Conv3D → MaxPool3D → BatchNorm] con 16 / 32 / 32 filtros → Flatten → Dense(64) → Dropout(0,3) → sigmoide

| Bloque | Parámetros |
|---|---|
| Bloque 1 | 512 |
| Bloque 2 | 13 984 |
| Bloque 3 | 27 808 |
| Cabeza (Flatten + Dense + salida) | 2 408 577 (98 %) |
| **Total** | **2 450 881** |

🖥️ Mostrar `model.summary()`.

**Qué decir**
- Fórmula: (k³·C_in + 1)·C_out. Ejemplo: (27·1 + 1)·16 = 448.
- El `Flatten` entrega 37 632 valores a `Dense(64)`: ahí está casi todo el modelo.

---

## 3. P1 · Predicción vs realidad (0:35)

**En la diapositiva**
- Predicción: sobreajuste, brecha ≥ 10 puntos, `val_loss` con mínimo y luego subida.
- Realidad (época 10): train 0,841 vs val 0,700 → brecha de 14 puntos ✔
- `val_loss` no mostró un mínimo claro en 10 épocas ✘
- Épocas 1 a 4: `val_acc` = 0,50 con `val_loss` 1,02–1,21 (no previsto)

🖥️ Mostrar las curvas de `show_results`.

**Qué decir**
- Lo de las épocas 1 a 4 no es sobreajuste: es una diferencia entre entrenamiento e inferencia. Hipótesis: BatchNorm con batch 2 (no comprobada).

---

## 4. P2 · Tabla de ablación (1:00)

| Variante | Parámetros | Val. acc | F1 | Mejor época | Observación |
|---|---|---|---|---|---|
| (a) Base densa | 2 450 881 | 0,700 | 0,743 | 7 | Sobreajuste incipiente |
| (b) Residual 3D | 3 324 193 | 0,500 | 0,000 | 1 | No aprendió (loss = 0,693) |
| (c) GAP | 42 337 | 0,667 | 0,737 | 8 | Aprende lento, 9 FP |

**Qué decir**
- Hipótesis previas: (a) sobreajusta; (b) mejora pequeña porque la red es poco profunda; (c) menos sobreajuste con riesgo de subajuste.
- Controles: mismos filtros, lr, épocas y callbacks.

---

## 5. P2 · Por qué cambian los resultados (1:15)

**En la diapositiva**
- **(b):** loss 2,07 → 0,693 = ln 2 → salida constante 0,5 → todo "normal". Se repitió en la corrida de 30 épocas.
  - Hipótesis: el bloque termina sin BatchNorm y `Dense(64)` recibe 50 400 valores sin normalizar.
  - No dice nada contra las residuales en sí (su ventaja es el flujo del gradiente en redes profundas).
- **(c):** GAP promedia cada mapa → 58 veces menos parámetros → sin sobreajuste, pero lento (`val_acc` = 0,50 hasta la época 7).
- **Elección:** (a), pero la diferencia con (c) es 1 error (21 vs 20) → ruido.

**Qué decir**
- Reportar el fallo de (b) tal cual era más honesto que ocultarlo.

---

## 6. P3 · Estrategia slice-wise (0:45)

**En la diapositiva**
- ResNet50 de ImageNet aplicada a 16 cortes (1 de cada 4)
- Gris → 3 canales, preprocesamiento de ResNet50 (0–255 menos la media de ImageNet)
- 2048 características por corte → promedio sobre la profundidad → Dropout → sigmoide
- BatchNorm del backbone siempre congelada

**Qué decir**
- Por qué slice-wise y no inflación 3D: reutiliza los pesos 2D sin reescribir kernels.

---

## 7. P3 · Feature extraction vs fine-tuning (1:00)

| Modelo | Entrenables | lr | Val. acc | F1 | Aciertos |
|---|---|---|---|---|---|
| Base desde cero (a) | 2 450 721 | 1e-4 | 0,700 | 0,743 | 21/30 |
| Feature extraction | 2 049 | 1e-3 | 0,867 | 0,867 | 26/30 |
| Fine-tuning (`conv5_`) | 14 955 521 | 1e-4 | **0,967** | **0,966** | 29/30 |

🖥️ Mostrar la celda que imprime `resultados`.

**Qué decir**
- Domain gap: ImageNet son fotos a color; aquí son cortes de TC en gris. Las características de bajo nivel se transfieren; el fine-tuning de `conv5_` adapta las de alto nivel.
- Reservas: una sola corrida, y la mejor época se eligió con el mismo test → 0,967 es optimista.
- BatchNorm: con batch 2 × 16 cortes de 2 pacientes, las estadísticas no son representativas → congeladas con las de ImageNet.

---

## 8. P4 · Grad-CAM (1:30)

**En la diapositiva**
- Grad-CAM sobre `conv3d_2` del modelo base (28×28×12 → reescalado)
- Por qué no CAM: la base no tiene GAP. Grad-CAM generaliza CAM usando el promedio de los gradientes como peso.

| Caso | p | Calor en borde | Dónde mira |
|---|---|---|---|
| Acierto anormal (idx 5) | 0,99 | 11 % | Pocos puntos en la periferia del pulmón |
| Acierto normal (idx 13) | 0,15 | 52 % | Difuso: pared, fondo, mesa |
| Error FP (idx 11) | 0,88 | 46 % | Pared, hombro, esternón |
| Error FP (idx 2) | 0,77 | 52 % | Pared y fuera del cuerpo |

🖥️ Mostrar la figura de Grad-CAM.

**Qué decir**
- El borde ocupa ≈51 % del área → 52 % de calor no prueba un sesgo a los bordes; indica calor sin foco.
- Limitación: no lo hice sobre el mejor modelo global (fine-tuning).

---

## 9. P5 · Fiabilidad y BatchNorm (1:00)

**En la diapositiva**

| Modelo | Acc | IC 95 % (Wilson) |
|---|---|---|
| Base | 0,700 | [0,521 – 0,833] |
| GAP | 0,667 | [0,488 – 0,808] |
| Feature extraction | 0,867 | [0,703 – 0,947] |
| Fine-tuning | 0,967 | [0,833 – 0,994] |

- Base y GAP no se distinguen; el fine-tuning apenas se separa de la base.
- BatchNorm con batch 2: estadísticas ruidosas en entrenamiento, medias móviles lentas (momentum 0,99) en inferencia → alternativa: GroupNorm / LayerNorm o acumulación de gradientes.

**Qué decir**
- Un error mueve la accuracy 3,3 puntos.

---

## 10. P5 · Métrica clínica y modo de fallo (1:00)

**En la diapositiva**

| Modelo | Recall | Especificidad | AUC | FN | FP |
|---|---|---|---|---|---|
| Base | 0,87 | 0,53 | 0,85 | 2 | 7 |
| GAP | 0,93 | 0,40 | 0,71 | 1 | 9 |
| Fine-tuning | 0,93 | 1,00 | 0,97 | 1 | 0 |

- En tamizaje, un falso negativo cuesta más → optimizar recall con especificidad aceptable; comparar con AUC.
- Modo de fallo: falsos positivos de alta confianza apoyados en pared/fondo.
- Corrección: máscara de pulmón (umbral HU + componentes conectados) y verificar de nuevo con Grad-CAM.

---

## 11. Cierre (0:30)

**En la diapositiva**
- Lo que no funcionó: residual (b), 10 épocas cortas para GAP, Grad-CAM no aplicado al mejor modelo.
- Uso de IA: Claude para generar código, depurar y redactar borradores; ejecución y decisiones propias.

**Qué decir**
- Una frase con la conclusión: el transfer learning con fine-tuning fue lo único que superó claramente a los modelos desde cero.

---

## Consejos para grabar

- Ten el notebook abierto en otra pestaña y cambia a él solo en las diapositivas marcadas 🖥️.
- Cronometra un ensayo: si pasa de 9:45, recorta primero las diapositivas 6 y 11.
- Preguntas que deberías poder responder sin leer:
  - ¿Por qué a la época 10 el lr está en ≈35 %? → 0,9^10 ≈ 0,35.
  - ¿Por qué el `Flatten` tiene 37 632 valores? → 14×14×6×32.
  - ¿Por qué ln 2 significa que no aprendió? → es la loss de predecir 0,5 siempre.
  - ¿Por qué congelaste BatchNorm en el fine-tuning?
