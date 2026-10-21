# Notas — Elhage et al. (2022), *Toy Models of Superposition*

Referencia: Elhage, N. et al. (2022). *Toy Models of Superposition*. arXiv:2209.10652.
Estado de lectura: informe estructurado vía alphaXiv (no leí el PDF completo; el PDF se guarda en esta carpeta para revisarlo antes de exponer).
Notas en palabras propias, sin citas literales.

## Idea central
- Un modelo con menos neuronas que rasgos útiles puede representar más rasgos que dimensiones si los rasgos son esparsos (rara vez aparecen a la vez). A eso lo llaman superposición.
- La consecuencia visible es la polisemanticidad: una neurona responde a varios rasgos sin relación entre sí.
- Es el mismo planteamiento de la lámina de la clase 10 (polisemanticidad y superposición).

## Montaje experimental (el que reproduzco en pequeño)
- Datos sintéticos: rasgos esparsos, con importancia variable.
- Modelo: comprime con una matriz W a pocas dimensiones y reconstruye con la transpuesta más una ReLU.
- Controles: esparsidad de los rasgos, importancia, razón rasgos/neuronas.

## Resultados que me sirven
- Hay un cambio de fase: con rasgos densos, cada rasgo importante recibe su dimensión; al aumentar la esparsidad, empiezan a compartir dimensiones.
- Los rasgos en superposición se acomodan en estructuras geométricas (pares opuestos, triángulos, pentágonos…) que minimizan la interferencia.
- La superposición se asocia a interferencia entre rasgos y, según los autores, podría contribuir a la vulnerabilidad adversaria.
- Sugieren estudiar direcciones de rasgos distribuidas en vez de neuronas individuales.

## Cómo lo uso en el cuaderno
- Hipótesis 1 (superposición): con esparsidad 0.5 el modelo no puede representar todos los rasgos; con 0.9 y 0.97 sí, mezclados.
- Mi modelo de juguete es una simplificación: 12 rasgos independientes, uniformes en importancia, 4 neuronas.

## Cautelas
- Es un modelo de juguete; los autores mismos no afirman que los modelos reales funcionen igual.
