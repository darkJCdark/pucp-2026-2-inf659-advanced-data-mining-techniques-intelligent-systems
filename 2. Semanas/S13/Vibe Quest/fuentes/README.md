# fuentes · Vibe Quest S13 (pregunta 2: Transformer sin Positional Encoding)

Regla acordada: solo fuentes indexadas en **Scopus o IEEE Xplore, de 2022 a 2026**. Los 6 PDF están en esta carpeta (descargados con acceso institucional PUCP). Leí sus textos con `pdftotext`: resúmenes, metodología y las secciones citadas.

| Fuente | Índice | Qué respalda en el Vibe Quest | Página |
| --- | --- | --- | --- |
| Zhang, Ge, Zhu, Ambikairajah, Song, Ni y Li (2024). *An Empirical Study on the Impact of Positional Encoding in Transformer-Based Monaural Speech Enhancement*. ICASSP 2024, pp. 1001-1005. DOI 10.1109/ICASSP48485.2024.10446337 | IEEE Xplore | Los Transformers son *position-agnostic*; el PE ayuda mucho en configuración no causal y poco en la causal, porque la máscara causal no es invariante a permutaciones (comparar bidireccional y causal) | Resumen p. 1001; máscara causal p. 1004 |
| Guo (2024). *Application and Impact of Position Embedding in Natural Language Processing*. ICEIB 2024, pp. 246-250. DOI 10.1109/ICEIB61477.2024.10602658 | IEEE Xplore | Misma configuración base, solo cambia el PE y NoPos es el control; el PE importa en tareas donde el orden es crucial y menos en las semánticas (tarea de orden frente a idioma) | Resumen p. 246; control p. 247 |
| Banđur, Banđur y Jakšić (2026). *Information-Theoretic Analysis of Positional Encoding Strategies in Vision Transformers*. IEEE TPAMI (versión aceptada). DOI 10.1109/TPAMI.2026.3736257 | IEEE Xplore | La self-attention es equivariante a permutaciones sin información de posición; al quitar el PE sinusoidal la exactitud cae casi al azar (ablación); 3 semillas por método | p. 1 (equivarianza); p. 8 (quitar el PE) |
| Hayakawa y Sato (2025). *Theoretical Analysis of Hierarchical Language Recognition and Generation by Transformers without Positional Encoding*. ACL 2025, pp. 30777-30834. DOI 10.18653/v1/2025.acl-long.1488 | Scopus | La máscara causal y un token de inicio permiten calcular la posición sin PE (uso de `<BOS>` y comparación causal) | Resumen p. 30777 |
| Zhao et al. (2024). *Length Extrapolation of Transformers: A Survey from the Perspective of Positional Encoding*. Findings of EMNLP 2024, pp. 9959-9977. DOI 10.18653/v1/2024.findings-emnlp.582 | Scopus | Sin PE los Transformers son equivalentes a permutaciones; los modelos causales pueden aprender posición sin PE | p. 9960 y nota 2 |
| Wang et al. (2024). *DeepNet: Scaling Transformers to 1,000 Layers*. IEEE TPAMI 46(10), 6761-6774. DOI 10.1109/TPAMI.2024.3386927 | IEEE Xplore y Scopus | Contexto de Add & Norm en la arquitectura usada | Resumen p. 6761 |

## Datos (no es fuente científica)
Tatoeba español-inglés, tatoeba.org vía manythings.org/anki, licencia CC-BY 2.0 (copia de TensorFlow).

## Material del curso
Clase 12 (láminas 120 y 122 y transcripción): PE sinusoidal, invarianza a permutaciones y Add & Norm.
