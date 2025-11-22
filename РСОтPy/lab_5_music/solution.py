import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Частоти нот (можна додати свої)
tone_map = {
    'C': 261.63,
    'D': 293.66,
    'E': 329.63,
    'F': 349.23,
    'G': 392.00,
    'A': 440.00,
    'B': 493.88
}

# Синтезатор однієї ноти
def tone_synthesizer(freq, duration, amplitude, sampling_freq):
    t = np.linspace(0, duration, int(sampling_freq * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    return signal

# Твоя мелодія
tone_sequence = [
    ('G', 0.4), ('E', 0.4), ('G', 0.4),
    ('A', 0.6), ('G', 0.4), ('E', 0.4),
    ('D', 0.6), ('C', 0.8)
]

sampling_freq = 44100   # стандартна аудіодискретизація
amplitude = 0.5         # гучність
signal = np.array([])

# Генерація мелодії
for tone_name, duration in tone_sequence:
    freq = tone_map[tone_name]       # знаходимо частоту
    synthesized = tone_synthesizer(freq, duration, amplitude, sampling_freq)
    signal = np.append(signal, synthesized)

# Нормалізація та конвертація у 16-бітний звук (WAV формат)
signal_int16 = np.int16(signal / np.max(np.abs(signal)) * 32767)

# Збереження у WAV
file_name = "melody.wav"
write(file_name, sampling_freq, signal_int16)

print("Готово! Файл записано як:", file_name)
