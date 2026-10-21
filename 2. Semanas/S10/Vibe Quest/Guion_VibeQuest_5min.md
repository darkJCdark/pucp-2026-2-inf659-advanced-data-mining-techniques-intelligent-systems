# Guion · Vibe Quest S10 · 5 minutos

**Regla:** solo digo lo que puedo defender. Todo número sale del cuaderno ejecutado.

## 0:00–0:40 · Pregunta y enfoque
"Mi pregunta es qué cambia si podemos leer y dirigir los rasgos internos de un modelo con autoencoders dispersos, y qué riesgos trae. No puedo entrenar un modelo de lenguaje aquí, así que reproduzco el mecanismo en un mundo sintético de 12 rasgos donde sé la respuesta correcta."

## 0:40–1:40 · Superposición (Panel A, esparsidad 0.97)
"Cuatro neuronas, doce rasgos. Como los rasgos son esparsos el modelo los comprime y las neuronas quedan mezcladas: pureza 0.42. Con esparsidad 0.5 ni siquiera cabe todo: solo 8 de 12."

## 1:40–2:50 · El SAE (Panel A, mover α)
"El SAE tiene 24 latentes y penalización L1. Con α = 0.3 los latentes son mucho más puros (0.89 frente a 0.42). Sin penalización recupera 25 %. Si me paso con α = 1 el error de reconstrucción sube a 0.48. En el mundo denso no ayuda."
"Una semilla da 100 %; el promedio de cinco es 80 % (±14)."

## 2:50–3:50 · Steering (Panel B)
"Fijo un latente al triple de su máximo y miro qué cambia. Con el SAE la selectividad promedio es 0.81; con la neurona, 0.23: en varias neuronas el rasgo objetivo ni se mueve. Pero incluso con el SAE hay efectos colaterales."

## 3:50–4:40 · Auditoría, alineación, sesgos, riesgos
"Auditar: se pueden leer rasgos como código inseguro o adulación (Templeton 2024), pero los autores advierten que existir el rasgo no prueba utilidad. Alinear: el rechazo es una dirección (Arditi 2024), y por eso también se apaga con facilidad. O'Brien 2024: reforzarla sube el sobre-rechazo y baja MMLU, TruthfulQA y GSM8K. AxBench: el prompting supera a los SAE; Jørgensen y Hansen 2026 lo matizan. Riesgos: doble uso, efectos colaterales, manipulación invisible, falsa seguridad."

## 4:40–5:00 · Cierre
"Funciona en pequeño y con ajuste fino; no demuestra que escale. Lo que sí queda: toda intervención necesita evaluar sus efectos colaterales."

## Cifras a la mano
| Cifra | Valor |
|---|---|
| Rasgos representados (esp. 0.5 / 0.9 / 0.97) | 8 / 11 / 12 de 12 |
| Pureza neuronas (0.5 / 0.9 / 0.97) | 0.55 / 0.45 / 0.42 |
| Pureza latentes, α = 0.3 (0.9 / 0.97) | 0.85 / 0.89 |
| Recupera α = 0 (0.9 / 0.97) | 0.27 / 0.25 |
| Recupera α = 0.3 semilla 0 | 1.00 |
| Recupera, 5 semillas | 0.80 ± 0.14 |
| Selectividad SAE / neurona, 5 semillas | 0.81 / 0.23 |
| Error relativo α = 0.3 / α = 1 | ≈ 0.10 / ≈ 0.48 |

## Preguntas probables
1. **¿Por qué α = 0.3?** Lo elegí explorando; con 0 y 1 el resultado cae. Es una limitación: está afinado en este mundo.
2. **¿Por qué 100 % en el panel y 80 % en la tabla?** Semilla 0 es el mejor caso; el promedio de cinco semillas es 0.80.
3. **¿Esto prueba que funciona en Claude?** No. Es un mundo sintético con rasgos conocidos.
4. **¿Qué es selectividad?** Cambio en el rasgo objetivo dividido por la suma del cambio en los demás; mayor que 1 significa más efecto en el objetivo que en el resto sumado.
5. **¿Qué es polisemanticidad?** Una neurona que responde a varios rasgos; surge porque el modelo comprime más rasgos que neuronas cuando son esparsos.
6. **¿Qué parte la hizo la IA?** Búsqueda y lectura asistida de fuentes, código y ejecución; yo definí el planteamiento, las métricas y revisé los resultados. Debo poder explicar cada línea del código.
