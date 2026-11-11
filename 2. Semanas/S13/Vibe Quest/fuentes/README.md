# fuentes · Vibe Quest S13 (pregunta 2: Transformer sin Positional Encoding)

Regla acordada: solo fuentes indexadas en **Scopus o IEEE Xplore, de 2022 a 2026**. La lista con DOI está en `fuentes_doi.csv` y el script `../codigo/descargar_fuentes.py` descarga los PDF abriendo cada DOI en un navegador real (con tu sesión institucional PUCP).

## Estado de lectura

| Fuente | Dónde está indexada | Qué respalda | PDF | Lectura |
| --- | --- | --- | --- | --- |
| Zhang, Ge, Zhu, Ambikairajah, Song, Ni y Li (2024). *An Empirical Study on the Impact of Positional Encoding in Transformer-Based Monaural Speech Enhancement*. ICASSP 2024, pp. 1001-1005. DOI 10.1109/ICASSP48485.2024.10446337 | IEEE Xplore | "Los Transformers son position-agnostic"; el PE permite distinguir el orden. Con atención no causal el PE ayuda de forma significativa; en configuración causal "no es muy útil", lo que sugiere que la atención causal puede incorporar la posición de forma implícita | Pendiente (script) | Resumen completo en IEEE Xplore |
| Guo (2024). *Application and Impact of Position Embedding in Natural Language Processing*. ICEIB 2024, pp. 246-250. DOI 10.1109/ICEIB61477.2024.10602658 | IEEE Xplore | Compara sinusoidal, RoPE, ALiBi y NoPos en generación de texto y análisis de sentimiento: ALiBi gana en generación; en sentimiento RoPE y NoPos rinden parecido | Pendiente (script) | Resumen completo |
| Banđur, Banđur y Jakšić (2026). *Information-Theoretic Analysis of Positional Encoding Strategies in Vision Transformers*. IEEE TPAMI. DOI 10.1109/TPAMI.2026.3736257 | IEEE Xplore | "La self-attention no tiene noción nativa del orden de los tokens"; usa PCA y una ablación que quita el PE | Pendiente (script) | Resumen completo |
| Hayakawa y Sato (2025). *Theoretical Analysis of Hierarchical Language Recognition and Generation by Transformers without Positional Encoding*. ACL 2025, pp. 30777-30834. DOI 10.18653/v1/2025.acl-long.1488 | Scopus | Con máscara causal y un token de inicio, un Transformer sin PE puede calcular información posicional (prueba constructiva). Es el límite de mi experimento (codificador bidireccional) | Pendiente (script) | Resumen en ACL Anthology |
| Zhao et al. (2024). *Length Extrapolation of Transformers: A Survey from the Perspective of Positional Encoding*. Findings of EMNLP 2024, pp. 9959-9977. DOI 10.18653/v1/2024.findings-emnlp.582 | Scopus | Marco: el PE como eje de la extrapolación a longitudes no vistas | Pendiente (script) | Resumen en ACL Anthology |
| Wang et al. (2024). *DeepNet: Scaling Transformers to 1,000 Layers*. IEEE TPAMI 46(10), 6761-6774. DOI 10.1109/TPAMI.2024.3386927 | IEEE Xplore y Scopus | Contexto de Add & Norm: modificar la conexión residual con una nueva normalización estabiliza Transformers extremadamente profundos | Pendiente (script) | Resumen completo |

## Qué no verifiqué
- Leí solo los **resúmenes**, no los textos completos: la sesión del navegador del chat no tiene acceso institucional a IEEE (el texto completo pide iniciar sesión). Por eso no cito páginas.
- Verifiqué en Scopus que aparecen Hayakawa y Sato, Zhao et al. y DeepNet. No comprobé la indexación en Scopus de Zhang, Guo y Banđur; sí están en IEEE Xplore, que basta para la regla.
- Los DOI de Hayakawa y Sato y de Zhao et al. los obtuve de Crossref y coinciden en título, año y páginas con lo que muestra Scopus.
- La referencia de este curso (Clase 12, láminas 120 y 122) no es fuente científica; solo plantea la pregunta.

## Descarga de los PDF
```bash
pip install playwright pandas && playwright install chromium
python ../codigo/descargar_fuentes.py
```
