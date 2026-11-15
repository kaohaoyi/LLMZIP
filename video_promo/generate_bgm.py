import numpy as np
import wave
import struct
import subprocess
import os

sample_rate = 44100
duration = 32.0  # seconds
bpm = 128.0
beat_len = 60.0 / bpm  # ~0.46875s
sixteenth = beat_len / 4.0

total_samples = int(sample_rate * duration)
t = np.linspace(0, duration, total_samples, endpoint=False)

# Track buffers
left = np.zeros(total_samples, dtype=np.float32)
right = np.zeros(total_samples, dtype=np.float32)

# 1. Kick Drum (every beat)
kick_samples = int(sample_rate * 0.25)
kt = np.linspace(0, 0.25, kick_samples, endpoint=False)
kick_freq = 140.0 * np.exp(-18.0 * kt) + 40.0
kick_env = np.exp(-12.0 * kt)
kick_wave = np.sin(2.0 * np.pi * kick_freq * kt) * kick_env
# Add initial click
click = np.random.uniform(-0.3, 0.3, int(sample_rate * 0.005)) * np.exp(-np.linspace(0, 10, int(sample_rate * 0.005)))
kick_wave[:len(click)] += click
kick_wave = np.clip(kick_wave, -1.0, 1.0) * 0.85

num_beats = int(duration / beat_len)
for b in range(num_beats):
    start_idx = int(b * beat_len * sample_rate)
    end_idx = min(start_idx + kick_samples, total_samples)
    seg_len = end_idx - start_idx
    if seg_len > 0:
        left[start_idx:end_idx] += kick_wave[:seg_len]
        right[start_idx:end_idx] += kick_wave[:seg_len]

# 2. Snare / Clap (beats 2 and 4)
snare_samples = int(sample_rate * 0.2)
st = np.linspace(0, 0.2, snare_samples, endpoint=False)
snare_noise = np.random.uniform(-0.5, 0.5, snare_samples) * np.exp(-14.0 * st)
snare_tone = np.sin(2.0 * np.pi * 180.0 * st) * np.exp(-20.0 * st) * 0.4
snare_wave = (snare_noise + snare_tone) * 0.65

for b in range(num_beats):
    if b % 4 in [1, 3]:  # beat 2 and 4
        start_idx = int(b * beat_len * sample_rate)
        end_idx = min(start_idx + snare_samples, total_samples)
        seg_len = end_idx - start_idx
        if seg_len > 0:
            left[start_idx:end_idx] += snare_wave[:seg_len]
            right[start_idx:end_idx] += snare_wave[:seg_len]

# 3. Hi-Hat (16th notes)
hat_samples = int(sample_rate * 0.05)
ht = np.linspace(0, 0.05, hat_samples, endpoint=False)
hat_wave = np.random.uniform(-0.3, 0.3, hat_samples) * np.exp(-40.0 * ht) * 0.35

total_sixteenths = int(duration / sixteenth)
for s in range(total_sixteenths):
    start_idx = int(s * sixteenth * sample_rate)
    end_idx = min(start_idx + hat_samples, total_samples)
    seg_len = end_idx - start_idx
    vol = 0.5 if s % 2 == 0 else 0.8
    if seg_len > 0:
        left[start_idx:end_idx] += hat_wave[:seg_len] * vol * 0.8
        right[start_idx:end_idx] += hat_wave[:seg_len] * vol * 1.2

# 4. Rolling Cyber Bass (8th notes, A minor chord progression)
# Roots: A1 (55Hz), F1 (43.65Hz), C2 (65.4Hz), G1 (49Hz)
chords = [55.0, 43.65, 65.4, 49.0]
eighth = beat_len / 2.0
total_eighths = int(duration / eighth)
bass_samples = int(sample_rate * (eighth * 0.9))
bt = np.linspace(0, eighth * 0.9, bass_samples, endpoint=False)

for e in range(total_eighths):
    chord_idx = int(e / 16) % len(chords)
    freq = chords[chord_idx]
    # Octave jump on every 4th 8th note
    if e % 4 == 2:
        freq *= 2.0
    
    # Sawtooth with rich harmonics
    b_wave = (
        0.5 * np.sin(2.0 * np.pi * freq * bt) +
        0.3 * np.sin(2.0 * np.pi * 2.0 * freq * bt) +
        0.15 * np.sin(2.0 * np.pi * 3.0 * freq * bt)
    )
    b_env = np.exp(-6.0 * bt)
    b_wave *= b_env * 0.55

    start_idx = int(e * eighth * sample_rate)
    end_idx = min(start_idx + bass_samples, total_samples)
    seg_len = end_idx - start_idx
    if seg_len > 0:
        left[start_idx:end_idx] += b_wave[:seg_len]
        right[start_idx:end_idx] += b_wave[:seg_len]

# 5. Arp / Lead (16th notes high synth)
# A Minor pentatonic: A4 (440), C5 (523.25), D5 (587.33), E5 (659.25), G5 (783.99)
notes = [440.0, 523.25, 659.25, 783.99, 659.25, 587.33, 523.25, 440.0]
lead_samples = int(sample_rate * (sixteenth * 0.85))
lt = np.linspace(0, sixteenth * 0.85, lead_samples, endpoint=False)

for s in range(total_sixteenths):
    freq = notes[s % len(notes)]
    l_wave = (np.sin(2.0 * np.pi * freq * lt) + 0.3 * np.sin(2.0 * np.pi * 2.0 * freq * lt)) * np.exp(-12.0 * lt) * 0.25
    start_idx = int(s * sixteenth * sample_rate)
    end_idx = min(start_idx + lead_samples, total_samples)
    seg_len = end_idx - start_idx
    pan = 0.5 + 0.3 * np.sin(s * 0.3)
    if seg_len > 0:
        left[start_idx:end_idx] += l_wave[:seg_len] * (1.0 - pan)
        right[start_idx:end_idx] += l_wave[:seg_len] * pan

# Master Limiter & Master Fade
# Fade in 0.5s, Fade out 2.0s
fade_in = np.minimum(1.0, np.linspace(0, 1.0, int(sample_rate * 0.5)))
fade_out = np.minimum(1.0, np.linspace(1.0, 0, int(sample_rate * 2.0)))
left[:len(fade_in)] *= fade_in
right[:len(fade_in)] *= fade_in
left[-len(fade_out):] *= fade_out
right[-len(fade_out):] *= fade_out

# Normalize to -1.0 dB
max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
if max_val > 0:
    left = (left / max_val) * 0.89
    right = (right / max_val) * 0.89

# Write WAV
wav_path = r"e:\Dev\AI\LLMZIP\video_promo\audio\bgm.wav"
mp3_path = r"e:\Dev\AI\LLMZIP\video_promo\audio\bgm.mp3"

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

print(f"Generated WAV: {wav_path}")
# Convert to MP3
subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-b:a", "192k", mp3_path], check=True)
print(f"Generated MP3: {mp3_path}")
