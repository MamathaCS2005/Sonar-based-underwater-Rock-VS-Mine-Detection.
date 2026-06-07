"""
Generate buzzer.wav alarm sound for the alert system.
Run this script once to create the audio file:
    python assets/generate_buzzer.py
"""
import struct
import wave
import math
import os


def generate_buzzer():
    """Generate a pulsing alarm tone WAV file."""
    output_path = os.path.join(os.path.dirname(__file__), "buzzer.wav")

    sample_rate = 44100
    duration = 1.5  # seconds
    frequency = 800  # Hz

    n_samples = int(sample_rate * duration)
    samples = []

    for i in range(n_samples):
        t = i / sample_rate
        # Create pulsing alarm tone (on/off pattern)
        envelope = 0.8 if (int(t * 4) % 2 == 0) else 0.3
        value = envelope * math.sin(2 * math.pi * frequency * t)
        # Add harmonic for richer alarm sound
        value += 0.3 * envelope * math.sin(2 * math.pi * frequency * 1.5 * t)
        samples.append(int(value * 32767 * 0.7))

    with wave.open(output_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for s in samples:
            wav_file.writeframes(struct.pack("<h", max(-32768, min(32767, s))))

    print(f"Buzzer audio generated: {output_path}")
    print(f"Duration: {duration}s | Sample rate: {sample_rate}Hz | Frequency: {frequency}Hz")


if __name__ == "__main__":
    generate_buzzer()
