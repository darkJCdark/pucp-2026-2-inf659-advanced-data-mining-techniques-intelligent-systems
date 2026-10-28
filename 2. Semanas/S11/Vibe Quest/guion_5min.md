# Guion de 5 minutos · Vibe Quest S11 (gradientes que se desvanecen y explotan)

Preparación antes de presentar: abrir el notebook en Colab, ejecutar todo (el entrenamiento de la sección 9 tarda entre 13 y 17 minutos) y dejar la pantalla en la sección 7. Las cifras que digo en voz alta salen de la pantalla; las de este guion son de mis ejecuciones y las tuyas pueden variar.

| Tiempo | Qué se muestra | Qué digo |
| --- | --- | --- |
| 0:00 a 0:30 | Título e hipótesis (inicio del notebook) | La pregunta es cómo afectan los gradientes que se desvanecen a una RNN simple cuando la respuesta depende de algo que pasó muchos pasos atrás, y si puedo demostrarlo con código frente a un modelo normal. Comparo la RNN simple con una LSTM. Mis tres hipótesis: el gradiente decae con la distancia, explota si el radio de la matriz recurrente es mayor que 1, y la LSTM lo conserva mejor. |
| 0:30 a 1:15 | Secciones 3, 4 y 6b | En la lámina 1.1^50 da 117 y 0.9^50 da 0.005, pero la fórmula es w^(T−1), o sea exponente 49; lo comprobé con autograd: 106.7 y 0.0057. Diseñé una tarea: la clase se muestra en el primer paso y los T−1 pasos siguientes son ruido; al final la red debe decir cuál fue. El azar es 25 %. Verifiqué con `assert` que mi recorrido manual coincide con `nn.RNN` y `nn.LSTM`. |
| 1:15 a 2:30 | Panel A (sección 7) | Sin entrenar nada, mido cuánto gradiente llega a cada paso. Con T = 50 y radio 1.0, el gradiente del primer paso es unas 0.05 veces el del último. Muevo el radio a 2.0: sube a unos 3 700, eso es explosión. Bajo el radio a 0.5 y T a 100: el gradiente se anula por completo en float32. Cambio a LSTM: queda cerca de 0.1 sin importar el radio. Fuente: la memoria depende de los autovalores de W_hh (Dou y Principe, 2023) y el efecto es más fuerte con tanh (Ceni, 2025). |
| 2:30 a 3:15 | Panel C (sección 11) | Entreno en vivo. RNN con T = 5: aprende. Subo T a 20 o 30 y pulso Entrenar; si la exactitud queda cerca de 0.25, muestro el log10 de la razón de gradientes muy negativo. Cambio a LSTM y repito. Si la RNN aprende con esa semilla, lo digo: el resultado depende de la semilla. |
| 3:15 a 4:15 | Gráfico de la sección 9d y resumen de la sección 15 | Con T = 5 y 10 todos aprenden. Con T ≥ 20 ni la RNN ni la LSTM aprenden siempre: cada modelo falla en alguna semilla, y la LSTM con la semilla 1 quedó en el azar en T = 50 y 80 en mis dos ejecuciones. Esto coincide con Rehmer y Kroll (2022): las unidades con compuertas no son inmunes y el entrenamiento depende de la inicialización. El recorte de gradiente controla la norma, pero no devolvió la memoria. |
| 4:15 a 4:45 | Sección 12 y sección 15b | Los chequeos automáticos marcan falso que la RNN falle de forma consistente con T = 50 y que la LSTM la supere de forma estable. Corrí el notebook dos veces con las mismas semillas y las cifras cambiaron: la RNN con T = 80 pasó de 0.64 a 0.25. Por eso no afirmo cuál modelo es mejor. |
| 4:45 a 5:00 | Conclusión | Lo que sí se observa: cuando la RNN falla, el gradiente que llega al primer paso es de 1e-5 a 1e-30 veces el del último, y el entrenamiento se vuelve dependiente de la semilla desde T = 20. Limitaciones: tarea sintética, 3 semillas, sin GRU. Harían falta más semillas. |

## Preguntas que pueden hacer

- **¿Por qué la LSTM también falla?** Rehmer y Kroll (2022) muestran que el gradiente también puede desvanecerse en unidades con compuertas; en mis corridas fallidas de la LSTM la razón de gradientes fue de 1e-4 a 1e-6. Además, la LSTM es sensible a la inicialización del sesgo de olvido: usé 2.0, apoyado en Tolić et al. (2024), que citan a Jozefowicz et al.
- **¿Por qué cambian los números entre ejecuciones?** Con las mismas semillas, PyTorch en CPU no da exactamente los mismos números en máquinas distintas y estas redes amplifican diferencias mínimas; no lo comprobé. Por eso reporto el patrón y no una cifra.
- **¿La tarea viene de un artículo?** No, la diseñé para aislar la dependencia a largo plazo; está indicado en la sección 4.
- **¿Por qué no GRU?** Lo acordé para mantener la comparación entre RNN simple y LSTM, que es lo que pide la pregunta 1.
- **¿Qué es el radio de W_hh?** La escala de la matriz recurrente ortogonal; con radio menor que 1 los autovalores contraen la información y con radio mayor que 1 la amplían.

## Qué probar antes de presentar

Ensayar una vez el Panel C con T = 20 y 600 pasos para saber cuánto tarda en tu máquina; no medí ese tiempo. Si el entrenamiento previo no terminó, el Panel B no funciona, así que conviene ejecutarlo con anticipación.
