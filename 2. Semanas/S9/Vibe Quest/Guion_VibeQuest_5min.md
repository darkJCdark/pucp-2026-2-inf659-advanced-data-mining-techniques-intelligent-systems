# Guion Vibe Quest S9 — pregunta 4 (5 minutos)

**Pregunta:** Explora el concepto de "manifold learning" en autoencoders y cómo se relaciona con PCA. ¿Cuáles son las implicaciones teóricas y prácticas de estas diferencias?

**Antes de exponer:** duración definida en 5 minutos. Subir `VibeQuest_S9_ManifoldLearning_PCA.ipynb` a un Colab nuevo y ejecutar todo (Entorno de ejecución > Ejecutar todo) para que los sliders respondan; no agregar celdas auxiliares. Si el tiempo se acorta, usar solo los bloques 1, 3 y 6 (marcados con ★).

## Bloques

| Min | Bloque | Qué hacer en pantalla |
|---|---|---|
| 0:00–0:30 ★ 1. Idea | Decir la tesis | Celda 2: rollo suizo en 3D |
| 0:30–1:15 2. Teoría | PCA y autoencoder lineal | Sin pantalla o tabla de la celda 5 |
| 1:15–2:30 ★ 3. Panel A | Mostrar la diferencia | Panel A: k=2, ruido 0, alternar AE lineal / no lineal |
| 2:30–3:15 4. Panel B | Qué aprende el latente | Panel B: PCA vs AE no lineal |
| 3:15–4:00 5. Costo y Isomap | Matices | Celdas 5b y 5c |
| 4:00–5:00 ★ 6. Cierre | Implicaciones | Sin pantalla |

## Texto sugerido (decirlo con tus palabras, no leerlo)

**1. Idea.** "Un manifold es una superficie de pocas dimensiones sobre la que viven los datos, aunque estén dentro de un espacio mayor. Mi ejemplo es una hoja de papel, que es 2D, enrollada en 3D. PCA y el autoencoder son dos formas de comprimir esos datos, y la diferencia está en qué tipo de superficie pueden representar."

**2. Teoría.** "PCA es la mejor proyección lineal: minimiza el error cuadrático de reconstrucción. Un autoencoder lineal, sin activaciones, entrenado con el mismo error, aprende el mismo subespacio. Esto lo demostraron Baldi y Hornik en 1989, y es lo que vimos en S9. Los pesos no son idénticos a los componentes principales, porque el latente no es único, pero el subespacio es el mismo. Si le agrego una activación no lineal, como tanh, la red puede doblarse y seguir una superficie curva. Bengio y colaboradores presentan a PCA como el caso lineal de la hipótesis del manifold."

**3. Panel A.** "Con k=2 y sin ruido, PCA y el autoencoder lineal dan el mismo error, 0.305. El autoencoder no lineal da 0.054 en la ejecución de la semilla 0, una mejora de 82 %. Con k=1 la mejora es de 47 %. Con k=3 no hay compresión, porque el rollo ya es 3D, y todos dan error cercano a cero. Para no depender de una semilla, repetí el experimento con 10 semillas: el no lineal ganó 10 de 10 en las seis configuraciones, con mejoras medias entre 37 % y 79 %." *(Mover el slider de ruido: la ventaja baja de 82 % a 65 % con ruido 0.6, porque parte del error ya no es reducible.)*

**4. Panel B.** "Aquí se ve la superficie que reconstruye cada método. PCA y el autoencoder lineal dan un plano. El no lineal da una hoja curva que envuelve el rollo. Lo que no hace es desenrollarlo: los colores en el espacio latente quedan mezclados. Entonces lo que puedo afirmar es que reconstruye mejor, no que haya aprendido la forma desenrollada."

**5. Costo e Isomap.** "En tiempo, PCA se ajusta en milisegundos y el autoencoder en 2 a 6 segundos: cientos a miles de veces más, según la corrida. Fournier y Aloise (2019) también encuentran que PCA es mucho más rápido y que los autoencoders ganan sobre todo con pocas dimensiones. Comparé además con Isomap, un método clásico de manifold learning: sin ruido sí desenrolla el rollo (correlación 1.00 con la posición en la hoja, frente a 0.2 de los demás), pero con ruido 0.3 cae a 0.20. Además no tiene decodificador, así que no mide error de reconstrucción ni se aplica de forma directa a puntos nuevos, que es el punto de Bengio."

**6. Cierre.** "Teóricamente: PCA es el caso lineal; el autoencoder no lineal generaliza a variedades curvas, a cambio de perder la solución única y cerrada. Prácticamente: si los datos son casi lineales o necesito rapidez o interpretabilidad, uso PCA; si hay estructura curva, pocas dimensiones latentes y tengo datos y tiempo para entrenar, el autoencoder compensa. Los límites de mi trabajo: datos sintéticos de 3 dimensiones, una arquitectura y unos hiperparámetros fijos, y el rollo suizo es una elección didáctica, no viene de las fuentes."

## Cifras verificadas (para tener a mano)

| Dato | Valor | Origen |
|---|---|---|
| Error k=2, ruido 0 (semilla 0) | PCA 0.3049, AE lineal 0.3049, AE no lineal 0.0537 | `resultados/datos/resultados_panelA.csv` |
| Error k=1, ruido 0 (semilla 0) | PCA 0.6239, AE lineal 0.6243, AE no lineal 0.3291 | ídem |
| 10 semillas, k=1 | mejora media 51 % / 47 % / 37 % (ruido 0 / 0.3 / 0.6) | `resultados/datos/resultados_semillas_pytorch_resumen.csv` |
| 10 semillas, k=2 | mejora media 79 % / 68 % / 57 % | ídem |
| Gana el no lineal | 10 de 10 semillas en las 6 configuraciones | ídem |
| Diferencia máxima AE lineal vs PCA | 7.5 % en una corrida (k=1, ruido 0.6) | ídem |
| Tiempo k=2 | PCA 0.001–0.008 s; AE lineal 2.3–4.5 s; AE no lineal 4.9–6.0 s (varía entre corridas) | `resultados/datos/resultados_5b_5c_colab.csv` y celda 5b |
| Isomap, ruido 0 | Spearman 1.00; confiabilidad 0.999 | ídem |
| Isomap, ruido 0.3 | Spearman 0.20; confiabilidad 0.878 | ídem |
| Confiabilidad ruido 0: PCA / AE lineal / AE no lineal | 0.860 / 0.862 / 0.973 | ídem |

Los tiempos dependen de la máquina y de que las redes corren en CPU; decir el orden de magnitud (milisegundos frente a segundos), no el decimal.

## Preguntas probables y respuesta corta

1. **¿Qué es un manifold?** Una superficie de pocas dimensiones donde se concentran los datos. Mi ejemplo: una hoja 2D enrollada en 3D.
2. **¿Por qué el autoencoder lineal da lo mismo que PCA?** Porque sin activaciones solo puede hacer una proyección lineal, y la mejor posible con error cuadrático es el subespacio de PCA (Baldi y Hornik, 1989). Una composición de funciones lineales sigue siendo lineal (S3).
3. **Si es el mismo subespacio, ¿por qué los pesos no son los componentes principales?** Porque el latente no es único: cualquier transformación invertible del latente da la misma reconstrucción. Los ejes no vienen ordenados por varianza ni son ortogonales.
4. **¿Por qué el no lineal reconstruye mejor?** Porque tanh permite que la superficie de reconstrucción se doble y siga el rollo; un plano no puede.
5. **¿Por qué con k=3 todos dan cero?** Porque los datos ya tienen 3 dimensiones: no hay compresión, solo una reconstrucción de la identidad.
6. **¿Por qué baja la ventaja con ruido?** El ruido agrega varianza en direcciones que ningún método de baja dimensión reconstruye; el error irreducible sube y se achica la diferencia.
7. **¿El latente aprendió el manifold desenrollado?** No. Lo medido es mejor reconstrucción. La correlación del latente con la posición en la hoja es baja (0.20 a 0.25). Isomap sí desenrolla sin ruido.
8. **¿Por qué comparar con Isomap?** Es el método clásico de manifold learning y el rollo suizo es su ejemplo estándar (Tenenbaum et al., 2000). Permite separar "reconstruir mejor" de "desenrollar".
9. **¿Cuándo conviene PCA y cuándo el autoencoder?** PCA: datos casi lineales, rapidez, interpretabilidad, prototipado. Autoencoder: estructura curva, pocas dimensiones latentes, suficientes datos y cómputo. Fournier y Aloise (2019): ganancia mayor con pocas dimensiones y PCA ~100× más rápido en sus datos.
10. **¿Por qué PyTorch?** Es lo que se usa en clase. Todos los resultados (Panel A, 10 semillas, tiempo e Isomap) son de ese notebook.
11. **¿Por qué tanh y esa arquitectura (3→32→k→32→3)?** Es una elección de diseño, no optimizada con validación formal. Es una limitación declarada.
12. **¿Por qué estos hiperparámetros (Adam 0.005, 2500 épocas)?** Los elegí con un prototipo con semilla 0 y los mantuve en las 10 semillas. No hay conjunto de validación separado.
13. **¿Usaste IA?** Sí, para investigar, escribir y probar el código; declarado en el notebook. Verifiqué los resultados ejecutando el código en Colab y leí Baldi y Hornik y Bengio. Puedo explicar cada celda.
14. **¿Qué haría con más tiempo?** Datos reales (MNIST), medir si el latente sirve para clasificar, probar un autoencoder variacional.
15. **¿Los datos vienen de las fuentes?** No. Son sintéticos y didácticos. Fournier y Aloise usan MNIST, Fashion-MNIST y CIFAR-10; Wetzel usa modelos físicos (Ising, XY).

## Relación con los temas del curso
- **S3:** funciones de activación; composición de lineales sigue siendo lineal.
- **S4:** Adam y tasa de aprendizaje.
- **S9:** PCA, autoencoder, equivalencia lineal; VAE solo como concepto (no se afirma nada de él en la exposición).

## Riesgos al defender
- No decir "el autoencoder aprendió el manifold": decir "reconstruye mejor".
- No decir que el AE lineal "es exactamente" PCA en cada corrida: en una corrida difirió 7.5 %.
- No afirmar que las fuentes usan el rollo suizo.
- Las cifras de tiempo son de Colab T4 con redes en CPU.
