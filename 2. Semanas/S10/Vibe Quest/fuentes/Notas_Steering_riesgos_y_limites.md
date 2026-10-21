# Notas — Steering: eficacia, costos y riesgos (cinco fuentes)

Notas en palabras propias. Estado de lectura de cada una al final.

## 1. Arditi et al. (2024) — *Refusal in Language Models Is Mediated by a Single Direction* (NeurIPS 2024, arXiv:2406.11717)
- En 13 modelos de chat de código abierto (1,8 B a 72 B parámetros), el rechazo de peticiones dañinas está mediado por una dirección lineal del flujo residual, obtenida como diferencia de medias entre activaciones de instrucciones dañinas e inofensivas.
- Quitar esa dirección apaga el rechazo; sumarla induce rechazo incluso en peticiones inofensivas.
- Proponen una modificación de pesos (ortogonalización) equivalente a quitar la dirección: un jailbreak de caja blanca barato, con poco efecto en otras capacidades (TruthfulQA es la excepción).
- Lectura para mi tema: es la prueba de que la "perilla" de seguridad también es un riesgo (quien la controla puede apagarla).

## 2. Wu et al. (2025) — *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders* (arXiv:2501.17148)
- Benchmark con Gemma-2 (2 B y 9 B) para comparar métodos de detección de conceptos y de steering.
- Hallazgo principal: en steering, el prompting y el ajuste fino superan a los SAE; los SAE rinden por debajo incluso de métodos supervisados simples (diferencia de medias, etc.). Sin selección supervisada, el SAE tiene mal rendimiento de detección.
- En su ablación, fijar (clamp) el rasgo rindió peor que sumar un vector.
- Lectura: la promesa del steering con SAE no es evidente frente a alternativas baratas.

## 3. O'Brien et al. (2024) — *Steering Language Model Refusal with Sparse Autoencoders* (EleutherAI y Microsoft, arXiv:2411.11296)
- Phi-3 Mini; SAE Top-k de 24 576 rasgos. Fijar un rasgo asociado al rechazo aumenta el rechazo de peticiones inseguras y resiste jailbreaks de varios turnos.
- Costo: sube el rechazo de peticiones seguras (sobre-rechazo) y degrada MMLU, TruthfulQA y GSM8K de forma sistemática; la degradación aparece también al fijar un rasgo no relacionado con seguridad (filosofía).
- Lectura: el rasgo no es perfectamente monosemántico y su amplificación tiene efectos colaterales amplios. Lo equivalente en mi cuaderno es la selectividad menor que 1 y el cambio en otros rasgos.

## 4. Li y Kasneci (2026) — *Safety Cost of Steering Vectors Is Separable and Reducible* (COLM 2026, arXiv:2608.08383)
- Los vectores de steering, aun los dirigidos a conceptos inofensivos, pueden degradar la seguridad del modelo (más cumplimiento de peticiones dañinas). Citan a Arditi et al. y trabajos previos que relacionan ese efecto con el solapamiento con la dirección de rechazo.
- Proponen quitar del vector un componente "dañino para la seguridad" que casi no aporta al efecto de steering buscado.
- Lectura: respalda que toda intervención necesita evaluar efectos sobre la seguridad.

## 5. Jørgensen y Hansen (2026) — *Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines* (arXiv:2605.31183)
- Respuesta parcial a AxBench: con una selección y etiquetado supervisado de rasgos, los SAE se acercan al rendimiento de LoRA en AxBench.
- Indican además evidencia de que la alta dispersión (L0 bajo) podría no ser crucial para el steering.
- Lectura: la evidencia sobre SAE y steering es mixta y sigue en movimiento; no conviene afirmar ni que sirven ni que no sirven.

## Estado de lectura
- Arditi, AxBench, O'Brien: informe estructurado vía alphaXiv (no leí el PDF completo).
- Li y Kasneci: abstract e introducción (el texto completo es largo; no lo leí entero).
- Jørgensen y Hansen: abstract e introducción.
- Fuentes de 2026: son preprints o trabajos recientes; no pude verificar su revisión por pares salvo lo que dice la propia fuente (COLM 2026 para Li y Kasneci).
