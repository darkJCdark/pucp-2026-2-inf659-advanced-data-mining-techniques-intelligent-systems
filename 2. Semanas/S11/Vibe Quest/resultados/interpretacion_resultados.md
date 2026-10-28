# Interpretación de resultados (S11)

**Qué muestran los datos.** Con secuencias cortas (T ≤ 10) la RNN simple y la LSTM resuelven la tarea. Desde T = 20 la RNN deja de aprender de forma consistente: solo 2 de 12 corridas con T ≥ 20 llegan a 0.95 de exactitud, y con T = 50 las tres semillas se quedan cerca del azar (0.25 a 0.27). En la RNN simple con T = 50 y 80 el gradiente que llega al primer paso es entre 2e-6 y 5e-12 veces el del último; en las corridas exitosas de la LSTM esa razón se mantiene entre 0.15 y 1.2.

**Desvanecimiento y explosión.** Sin entrenar, el cociente de gradientes entre el primer y el último paso depende del radio de Wh: 1.5e-7 (0.9), 7.4e-6 (1.0), 0.83 (1.5) y 364 (2.0). Es la misma aritmética de la lámina (w^T con w < 1 o > 1) más la saturación de tanh.

**Matices que debo poder defender.**
- La RNN no siempre falla: en T = 20 y T = 30 una semilla llegó a ≈ 1.0. El desvanecimiento hace el entrenamiento difícil y dependiente de la inicialización, no imposible.
- La LSTM tampoco es infalible: con T = 50 la semilla 1 quedó en 0.256. Rehmer y Kroll (IJCNN 2022) afirman en su resumen que el gradiente también se desvanece en unidades con compuertas.
- El recorte de gradiente controla la explosión (normas máximas de hasta ≈ 1400 en la RNN sin recorte, T = 20) pero no mejora la memoria a largo plazo: con T = 80 queda en 0.32, como la RNN sin recorte.
- El sesgo de olvido 2.0 de la LSTM es una decisión mía respaldada por Tolić et al. (IEEE Access 2024), quienes citan a Jozefowicz et al. para inicializarlo en 1 o 2 "para permitir el flujo del gradiente". En el ajuste con la semilla 100, con sesgo 1.0 la LSTM falló con lr 0.01 y 0.02.

**Lo que no puedo concluir.** La tarea es sintética y la diseñé yo; 3 semillas dan estimaciones ruidosas; la GRU no se evaluó; los hiperparámetros se ajustaron con la misma tarea. Cinco de los seis artículos de IEEE los leí solo hasta el resumen; solo Tolić et al. está leído en el PDF.


**Nota tras ejecutar la versión PyTorch en Colab.** Con PyTorch, la RNN no falla de forma consistente con T ≥ 50 (media 0.64 en T = 50 y 80; una semilla llega a 1.0 con T = 80) y la LSTM falla con la semilla 1 desde T = 20. Lo que se sostiene en ambas implementaciones: el gradiente se desvanece cuando la RNN falla y el entrenamiento depende de la semilla. La frase "falla de forma consistente desde T ≥ 50" solo vale para la ejecución en NumPy.
