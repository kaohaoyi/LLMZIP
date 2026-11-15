import numpy as np
import wave
import subprocess
import os

sample_rate = 44100
bpm = 84.0
beat_len = 60.0 / bpm  # ~0.7143s
bar_len = beat_len * 4.0  # ~2.857s
duration = 48.0  # seconds

total_samples = int(sample_rate * duration)
left = np.zeros(total_samples, dtype=np.float32)
right = np.zeros(total_samples, dtype=np.float32)

# 1. Warm Vinyl / Tape Ambient Noise (Subtle relaxing floor)
vinyl_noise = np.random.normal(0, 0.015, total_samples)
# Subtle crackle pops
pops = (np.random.random(total_samples) > 0.9997) * np.random.uniform(-0.08, 0.08, total_samples)
ambient = (vinyl_noise + pops) * 0.4
left += ambient
right += ambient

# 2. Warm Lo-Fi Rhodes / Electric Piano Chords
# Fmaj7 -> Em7 -> Dm7 -> Cmaj7
chords = [
    [174.61, 220.00, 261.63, 329.63],  # Fmaj7: F3, A3, C4, E4
    [164.81, 196.00, 246.94, 293.66],  # Em7:   E3, G3, B3, D4
    [146.83, 174.61, 220.00, 261.63],  # Dm7:   D3, F3, A3, C4
    [130.81, 164.81, 196.00, 246.94]   # Cmaj7: C3, E3, G3, B3
]

num_bars = int(duration / bar_len) + 1
chord_samples = int(sample_rate * (bar_len * 0.96))
ct = np.linspace(0, bar_len * 0.96, chord_samples, endpoint=False)

# Tremolo effect (4.5 Hz subtle sine)
tremolo = 1.0 + 0.15 * np.sin(2.0 * np.pi * 4.5 * ct)

for b in range(num_bars):
    chord = chords[b % len(chords)]
    chord_wave = np.zeros(chord_samples, dtype=np.float32)
    for freq in chord:
        # Rhodes tone: Fundamental + soft 2nd & 3rd harmonics + gentle decay
        h1 = np.sin(2.0 * np.pi * freq * ct)
        h2 = 0.3 * np.sin(2.0 * np.pi * 2.0 * freq * ct)
        h3 = 0.1 * np.sin(2.0 * np.pi * 3.0 * freq * ct)
        env = np.exp(-1.4 * ct)
        chord_wave += (h1 + h2 + h3) * env
    
    chord_wave = chord_wave * tremolo * 0.16
    start_idx = int(b * bar_len * sample_rate)
    end_idx = min(start_idx + chord_samples, total_samples)
    seg_len = end_idx - start_idx
    if seg_len > 0:
        left[start_idx:end_idx] += chord_wave[:seg_len] * 0.9
        right[start_idx:end_idx] += chord_wave[:seg_len] * 1.1

# 3. Soft Pillowy Kick (beats 1 and 3 with subtle human timing)
kick_samples = int(sample_rate * 0.3)
kt = np.linspace(0, 0.3, kick_samples, endpoint=False)
# Deep warm 60Hz drop to 35Hz
k_freq = 65.0 * np.exp(-10.0 * kt) + 32.0
k_wave = np.sin(2.0 * np.pi * k_freq * kt) * np.exp(-10.0 * kt) * 0.55

num_beats = int(duration / beat_len)
for b in range(num_beats):
    # Lo-fi kick pattern: beat 1, beat 3-and
    if b % 4 in [0, 2]:
        start_idx = int(b * beat_len * sample_rate)
        end_idx = min(start_idx + kick_samples, total_samples)
        seg_len = end_idx - start_idx
        if seg_len > 0:
            left[start_idx:end_idx] += k_wave[:seg_len]
            right[start_idx:end_idx] += k_wave[:seg_len]

# 4. Soft Rimshot / Snare (beats 2 and 4)
rim_samples = int(sample_rate * 0.12)
rt = np.linspace(0, 0.12, rim_samples, endpoint=False)
rim_tone = np.sin(2.0 * np.pi * 320.0 * rt) * np.exp(-25.0 * rt) * 0.25
rim_noise = np.random.normal(0, 0.15, rim_samples) * np.exp(-28.0 * rt)
rim_wave = (rim_tone + rim_noise) * 0.45

for b in range(num_beats):
    if b % 4 in [1, 3]:
        start_idx = int(b * beat_len * sample_rate)
        end_idx = min(start_idx + rim_samples, total_samples)
        seg_len = end_idx - start_idx
        if seg_len > 0:
            left[start_idx:end_idx] += rim_wave[:seg_len]
            right[start_idx:end_idx] += rim_wave[:seg_len]

# 5. Delicate Lo-Fi Shaker / Hat (8th notes with swing)
eighth = beat_len / 2.0
num_eighths = int(duration / eighth)
shaker_samples = int(sample_rate * 0.06)
sht = np.linspace(0, 0.06, shaker_samples, endpoint=False)
shaker_wave = np.random.normal(0, 0.1, shaker_samples) * np.exp(-45.0 * sht) * 0.2

for e in range(num_eighths):
    start_idx = int(e * eighth * sample_rate)
    end_idx = min(start_idx + shaker_samples, total_samples)
    seg_len = end_idx - start_idx
    vel = 0.5 if e % 2 == 0 else 0.8
    if seg_len > 0:
        left[start_idx:end_idx] += shaker_wave[:seg_len] * vel * 0.85
        right[start_idx:end_idx] += shaker_wave[:seg_len] * vel * 1.15

# 6. Smooth Warm Bass (Roots: F2=87.31, E2=82.41, D2=73.42, C2=65.41)
bass_roots = [87.31, 82.41, 73.42, 65.41]
bass_samples = int(sample_rate * (bar_len * 0.85))
bt = np.linspace(0, bar_len * 0.85, bass_samples, endpoint=False)

for b in range(num_bars):
    freq = bass_roots[b % len(bass_roots)]
    # Soft sine + subtle second harmonic
    bass_wave = (np.sin(2.0 * np.pi * freq * bt) + 0.2 * np.sin(2.0 * np.pi * 2.0 * freq * bt)) * np.exp(-1.5 * bt) * 0.45
    start_idx = int(b * bar_len * sample_rate)
    end_idx = min(start_idx + bass_samples, total_samples)
    seg_len = end_idx - start_idx
    if seg_len > 0:
        left[start_idx:end_idx] += bass_wave[:seg_len]
        right[start_idx:end_idx] += bass_wave[:seg_len]

# Master Fade in (1.0s) & Fade out (2.5s)
fade_in = np.minimum(1.0, np.linspace(0, 1.0, int(sample_rate * 1.0)))
fade_out = np.minimum(1.0, np.linspace(1.0, 0, int(sample_rate * 2.5)))
left[:len(fade_in)] *= fade_in
right[:len(fade_in)] *= fade_in
left[-len(fade_out):] *= fade_out
right[-len(fade_out):] *= fade_out

# Peak normalize to -1.5 dB
max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
if max_val > 0:
    left = (left / max_val) * 0.82
    right = (right / max_val) * 0.82

wav_path = r"e:\Dev\AI\LLMZIP\video_promo\audio\bgm_focus.wav"
mp3_path = r"e:\Dev\AI\LLMZIP\video_promo\audio\bgm_focus.mp3"

left_int16 = (left * 32767).astype(np.int16)
right_int16 = (right * 32767).astype(np.int16)
stereo_interleaved = np.empty((total_samples * 2,), dtype=np.int16)
stereo_interleaved[0::2] = left_int16
stereo_interleaved[1::2] = right_int16

with wave.open(wav_path, "wb") as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(stereo_interleaved.tobytes())

subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-b:a", "192k", mp3_path], check=True)
print(f"Generated Relaxing Focus BGM: {mp3_path}")
