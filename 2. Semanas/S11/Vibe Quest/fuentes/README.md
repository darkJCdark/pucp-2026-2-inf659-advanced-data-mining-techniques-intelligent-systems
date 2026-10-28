# fuentes · Vibe Quest S11 (gradientes que se desvanecen y explotan)

Regla acordada: todas las fuentes deben estar indexadas en **Scopus o IEEE Xplore y ser de 2022 a 2026**. Las obras clásicas (Bengio 1994, Hochreiter y Schmidhuber 1997, Pascanu 2013, Jozefowicz 2015) solo se mencionan como "citado en" un artículo del rango.

La lista con DOI está en `fuentes_doi.csv`. El script `../codigo/descargar_fuentes.py` descarga los PDF abriendo cada DOI en un navegador real (requiere tu sesión institucional PUCP para los que no son de acceso abierto).

## Estado de lectura

| Fuente | Cómo se usa | PDF | Lectura |
| --- | --- | --- | --- |
| Tolić, Mileva Boshkoska y Skansi (2024). *Chrono Initialized LSTM Networks With Layer Normalization*. IEEE Access 12, 115220-115238. DOI 10.1109/ACCESS.2024.3445329 | Respaldo del sesgo de olvido: informa que Jozefowicz et al. sugieren inicializarlo en 1 o 2 "para permitir el flujo del gradiente" (p. 115221) | Descargado (acceso abierto) | Leído en las secciones sobre inicialización de compuertas |
| He, Mao y Yi (2022). *Subtraction Gates*. IEEE TNNLS. DOI 10.1109/TNNLS.2020.3043752 | El problema de gradiente desvaneciente/explosivo limita las dependencias largas; las compuertas de la LSTM las mitigan | Descargado | Leído en la introducción y las secciones citadas |
| Dou y Principe (2023). *Dynamic Analysis and an Eigen Initializer for RNNs*. IJCNN. DOI 10.1109/IJCNN54540.2023.10191986 | Análisis del estado oculto con la descomposición espectral de Wh (justifica variar el radio de Wh) | Descargado | Leído en la introducción y las secciones citadas |
| Ceni (2025). *Random Orthogonal Additive Filters*. IEEE TNNLS. DOI 10.1109/TNNLS.2025.3538924 | Valores singulares del jacobiano entrada-salida cerca de 1 | Descargado | Leído en la introducción y las secciones citadas |
| Rehmer y Kroll (2022). *The effect of the forget gate on bifurcation boundaries...*. IJCNN. DOI 10.1109/IJCNN55064.2022.9892458 | Afirma que el gradiente también se desvanece en unidades con compuertas; respalda que una LSTM puede fallar | Descargado | Leído en la introducción y las secciones citadas |
| Shan et al. (2023). *DRRNets*. IEEE TNNLS. DOI 10.1109/TNNLS.2021.3105818 | Dificultad de entrenar RNN en secuencias largas | Descargado | Leído en la introducción y las secciones citadas |

Los seis PDF están en esta carpeta. Se leyeron las partes citadas en el notebook (resúmenes, introducciones y las secciones indicadas en la sección 14 del `.ipynb`), no cada artículo completo. Páginas confirmadas: Ceni p. 10794 (introducción), Tolić p. 115221. Para Dou y Principe, Rehmer y Kroll, He y Shan se cita la sección, no la página.

## Material del curso (no es fuente científica)
Clase 11 (láminas y transcripción): ecuación ∂hT/∂h1 = w^(T−1) con los ejemplos 1.1^50 ≈ 117.4 y 0.9^50 ≈ 0.005, recorte de gradiente, LSTM y GRU.
