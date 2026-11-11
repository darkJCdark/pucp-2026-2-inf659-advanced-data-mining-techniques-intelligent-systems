# Vibe Quest S13 · Preguntas y lo visto en clase

Curso INF659 Técnicas Avanzadas de Data Mining y Sistemas Inteligentes (PUCP 2026-2). Clase 12, "Modelos Autorregresivos y Transformadores" (prof. Rodrigo Uribe). Las preguntas están en la lámina 123 de `Clase12.pdf` ("Preguntas del Vibe Quest") y se pide elegir una.

## Las 4 preguntas

1. Un compañero dice: "La Self-Attention es básicamente lo mismo que la atención de Bahdanau, solo más rápida." ¿Qué le corriges? ¿Puedes implementar ambos mecanismos en PyTorch y visualizar sus matrices de atención sobre la misma oración para mostrar en qué difieren?
2. ¿Qué le pasaría al Transformer si eliminamos el Positional Encoding y le damos las frases "El gato persiguió al perro" y "El perro persiguió al gato"? ¿Puedes demostrarlo con código comparando los embeddings de salida con y sin codificación posicional, y visualizando por qué el orden se pierde?
3. En el Decoder del Transformer hay dos tipos de atención por capa: Masked Self-Attention y Cross-Attention. ¿Puedes explicar de dónde vienen las Q, K y V en cada caso y diseñar una visualización que muestre qué tokens atiende el decoder en cada tipo durante la generación de una traducción?
4. ¿Por qué el bloque Add & Norm aparece después de cada subcapa del Transformer? ¿Puedes entrenar un Transformer con y sin conexiones residuales sobre una tarea de secuencia simple y graficar las curvas de pérdida para demostrar qué problema concreto resuelven al escalar a 6 o más capas?

## Lo visto en clase, por pregunta

Las páginas son de `Clase12.pdf`; "transcripción" es `DM S13.txt`.

### Pregunta 1 · Self-Attention frente a Bahdanau
- Bahdanau (p. 31 a 43): codificador con RNN bidireccional que crea una anotación por palabra; el decodificador, también una RNN, recibe en cada paso un vector de contexto que es una suma ponderada de las anotaciones. La atención es aditiva: `score = v_a · tanh(W_a [s; h])`, un MLP extra por cada par. Direccionamiento por contenido, no por posición (p. 35).
- Comparación Bahdanau y Luong (p. 43): Luong usa producto punto, tiene menor costo y mejor paralelismo en GPU; la lámina lo presenta como base del Scaled Dot-Product Attention.
- Costo de la atención en RNN (p. 40 a 42): el contexto se recalcula en cada paso (costo `t·k²` por paso) y sigue habiendo recurrencia; la ruta entre entrada y salida es constante.
- Self-Attention (p. 107): las consultas, claves y valores salen de los mismos tokens; la matriz de atención es `softmax(QKᵀ/√d)` y se multiplica por V. En la transcripción el profesor la describe como "una conversación entre llaves y consultas" y destaca que la matriz resultante es asimétrica.
- Tabla RNN y Transformer (p. 121): paralelismo, dependencias largas, cuello de botella y costo O(n²·d) por capa.

### Pregunta 2 · Sin Positional Encoding
- La Self-Attention es invariante al orden (p. 120): se suma un vector de posición al embedding, `input = Embedding(token) + PE(pos)`, con `PE(pos,2i) = sin(pos/10000^(2i/d))` y `PE(pos,2i+1) = cos(pos/10000^(2i/d))`.
- En la transcripción (hacia 02:38 a 02:42) el profesor explica que los tokens "miran a todos los tokens al mismo tiempo" y que, sin posición, "podría cambiar este aquí, este aquí, y los valores cambiarían, pero sería lo mismo". Lo compara con representar posiciones en binario, extendido de forma continua con senos y cosenos, y comenta que hoy hay otras formas de hacerlo.

### Pregunta 3 · Masked Self-Attention y Cross-Attention
- Cross-Attention (p. 111 y transcripción 02:28): "la matriz de queries la consigue de otra parte", es decir, los tokens de un lado atienden a los de otro. La lámina enlaza un video explicativo.
- Arquitectura completa del Transformer (p. 119) y resumen (p. 122): codificador de 6 capas y decodificador de 6 capas, Multi-Head Attention con h = 8, red feed-forward y Add & Norm.
- Multi-Head (p. 112): cada cabeza se especializa en algo distinto y una capa lineal decide cómo combinarlas.
- Nota de clase: la lámina y la transcripción no desarrollan la máscara causal del decodificador; eso hay que sustentarlo con fuentes.

### Pregunta 4 · Add & Norm y conexiones residuales
- Resumen (p. 122, punto 6): "LayerNorm + Residuals: permiten entrenar redes muy profundas estabilizando gradientes. Patrón: LayerNorm(x + Sublayer(x))".
- En la transcripción el profesor lo deja como pregunta para indagar (02:38): "¿por qué utilizamos Layer Normalization en los transformadores?".
- Conexión con S11: el problema de gradientes que se desvanecen y explotan de la Clase 11 (vanishing gradients, RNN frente a LSTM).

## Estructura de trabajo (igual a S9, S10 y S11)

```
S13/Vibe Quest/
├── codigo/
├── fuentes/            (solo Scopus o IEEE, 2022-2026)
├── resultados/
│   ├── capturas/
│   ├── datos/
│   └── google colab/
├── pipeline.md
├── README.md
├── guion_5min.md
└── vibe_quest_s13.ipynb
```

La carpeta `S13/Vibe Quest` ya existe con esa estructura y los archivos vacíos. Presentación de 5 minutos con sliders, con human-in-the-loop y el `.ipynb` solo al final.
