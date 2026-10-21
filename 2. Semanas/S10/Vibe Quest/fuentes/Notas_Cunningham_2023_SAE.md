# Notas — Cunningham et al. (2023), *Sparse Autoencoders Find Highly Interpretable Features in Language Models*

Referencia: Cunningham, H., Ewart, A., Riggs, L., Huben, R. y Sharkey, L. (2023). arXiv:2309.08600.
Estado de lectura: informe estructurado vía alphaXiv (no leí el PDF completo).
Notas en palabras propias.

## Qué hacen
- Entrenan autoencoders dispersos (SAE) sobre activaciones internas de modelos de lenguaje pequeños (Pythia-70M y Pythia-410M) para separar rasgos superpuestos.
- Arquitectura: una capa oculta con ReLU y más unidades que la dimensión de entrada (diccionario sobrecompleto); pesos atados en el caso principal.
- Pérdida: error de reconstrucción (L2) más penalización L1 sobre las activaciones del diccionario. Es la pérdida que uso en el cuaderno, a escala diminuta.

## Resultados
- Los rasgos del SAE son más interpretables que neuronas, direcciones aleatorias, PCA e ICA, según una métrica automática de interpretabilidad (usa GPT-4 para describir y GPT-3.5 para simular).
- En una tarea de atribución causal (identificación del objeto indirecto), los rasgos del SAE localizan mejor el comportamiento que PCA: hacen falta menos rasgos y menor magnitud de edición.
- El efecto desaparece sin penalización de dispersión (α = 0). Es el mismo patrón que observo yo en la tabla del cuaderno.

## Limitaciones declaradas
- La reconstrucción no es perfecta (sube la perplejidad al sustituir activaciones).
- Muchos rasgos "muertos" en algunas capas; la interpretación automática es imperfecta.

## Cómo lo uso
- Justifica el diseño del SAE del cuaderno (diccionario sobrecompleto, L1) y la comparación con una línea base (aquí, las neuronas).
