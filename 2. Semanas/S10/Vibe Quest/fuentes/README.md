# Fuentes — Vibe Quest S10 (pregunta 3: SAE, monosemanticidad y steering)

## Notas de lectura (en esta carpeta)

| Archivo | Cubre |
|---|---|
| `Notas_Elhage_2022_Superposicion.md` | Superposición y polisemanticidad en modelos de juguete |
| `Notas_Cunningham_2023_SAE.md` | Autoencoders dispersos (L2 + L1) sobre modelos de lenguaje |
| `Notas_Anthropic_2023_2024_Monosemanticidad.md` | Monosemanticidad, escala a Claude 3 Sonnet, steering (Golden Gate) y rasgos de seguridad |
| `Notas_Steering_riesgos_y_limites.md` | Arditi 2024, AxBench 2025, O'Brien 2024, Li y Kasneci 2026, Jørgensen y Hansen 2026 |

Las notas están en palabras propias, sin citas literales. No sustituyen leer el PDF antes de exponer.

## Fuentes y enlaces

| Fuente | Enlace | Estado de lectura | PDF en esta carpeta |
|---|---|---|---|
| Elhage et al. (2022), *Toy Models of Superposition* | https://arxiv.org/abs/2209.10652 | Informe vía alphaXiv | `Elhage_2022_Toy_Models_of_Superposition.pdf` (versión de Transformer Circuits Thread, no de arXiv) |
| Cunningham et al. (2023), *Sparse Autoencoders Find Highly Interpretable Features in Language Models* | https://arxiv.org/abs/2309.08600 | Informe vía alphaXiv | `Cunningham_2023_Sparse_Autoencoders.pdf` |
| Bricken et al. (2023), *Towards Monosemanticity* | https://transformer-circuits.pub/2023/monosemantic-features/index.html | Página leída ≈72 % (resumen asistido) | no hay PDF oficial |
| Templeton et al. (2024), *Scaling Monosemanticity* | https://transformer-circuits.pub/2024/scaling-monosemanticity/ | Página leída ≈78 % (resumen asistido) | no hay PDF oficial |
| Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction* | https://arxiv.org/abs/2406.11717 | Informe vía alphaXiv | `Arditi_2024_Refusal_single_direction.pdf` |
| Wu et al. (2025), *AxBench* | https://arxiv.org/abs/2501.17148 | Informe vía alphaXiv | `Wu_2025_AxBench.pdf` |
| O'Brien et al. (2024), *Steering Language Model Refusal with Sparse Autoencoders* | https://arxiv.org/abs/2411.11296 | Informe vía alphaXiv | `OBrien_2024_Steering_refusal_SAE.pdf` |
| Li y Kasneci (2026), *Safety Cost of Steering Vectors Is Separable and Reducible* (COLM 2026) | https://arxiv.org/abs/2608.08383 | Abstract e introducción | `Li_Kasneci_2026_Safety_cost_steering_vectors.pdf` |
| Jørgensen y Hansen (2026), *Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines* | https://arxiv.org/abs/2605.31183 | Abstract e introducción | `Jorgensen_Hansen_2026_SAE_vs_baselines.pdf` |
| Clase 10 (láminas y transcripción), curso INF659 | `2. Semanas/S10/Clase10.pdf` | Leída (láminas de polisemanticidad, SAE, Golden Gate) | ya está en la carpeta de la semana |

## Estado de descarga de PDF

Los 7 PDF de arXiv/Transformer Circuits están en esta carpeta y se comprobó que cada archivo empieza con `%PDF-` y que el título de la primera página coincide con la fuente (Elhage, Cunningham, Arditi, Wu, O'Brien, Li y Kasneci, Jørgensen y Hansen). Se descargaron con el navegador integrado desde `https://arxiv.org/pdf/<número>`.

Sin PDF oficial (son páginas web de Anthropic; para guardarlas, abrir el enlace e imprimir a PDF desde el navegador):
- Bricken et al. (2023): https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Templeton et al. (2024): https://transformer-circuits.pub/2024/scaling-monosemanticity/

Pendiente: leer los PDF completos (las notas de `Notas_*.md` salieron de informes y de abstract e introducción, no de la lectura completa).

## Cómo se verificó cada fuente

Cada referencia se localizó con alphaXiv (arXiv) o con la página del autor (Anthropic) y se leyó su resumen o informe antes de citarla. No se cita ninguna cifra que no aparezca en estas notas.
