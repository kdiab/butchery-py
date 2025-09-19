import numpy as np
import pyaudio
import threading
import time
from scipy import signal
from collections import deque
import librosa

class DrumDetector:
    def __init__(self, 
                 sample_rate=44100, 
                 chunk_size=1024, 
                 threshold=0.23,
                 input_device=None):
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.input_device = input_device
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        
        self.start_time = time.time()
        
        self.kick_holdoff = int(0.15 * sample_rate)  # 150ms
        self.snare_holdoff = int(0.1 * sample_rate)  # 100ms
        self.hihat_holdoff = int(0.06 * sample_rate) # 60ms
        
        self.last_kick_trigger = 0
        self.last_snare_trigger = 0
        self.last_hihat_trigger = 0
        
        self.sample_count = 0
        
        self.kick_env = 0.0
        self.snare_env = 0.0
        self.hihat_env = 0.0
        self.total_env = 0.0
        
        self.kick_attack = 1 - np.exp(-1 / (0.01 * sample_rate))
        self.kick_release = 1 - np.exp(-1 / (0.1 * sample_rate))
        self.snare_attack = 1 - np.exp(-1 / (0.01 * sample_rate))
        self.snare_release = 1 - np.exp(-1 / (0.1 * sample_rate))
        self.hihat_attack = 1 - np.exp(-1 / (0.001 * sample_rate))
        self.hihat_release = 1 - np.exp(-1 / (0.05 * sample_rate))
        self.total_attack = 1 - np.exp(-1 / (0.01 * sample_rate))
        self.total_release = 1 - np.exp(-1 / (0.1 * sample_rate))
        
        self.zcr_buffer_size = int(0.02 * sample_rate)  # 20ms
        self.audio_buffer = deque(maxlen=self.zcr_buffer_size)
        
        self.setup_filters()
        
        self.kick_filter_state = signal.lfilter_zi(self.kick_b, self.kick_a)
        self.snare_filter_state = signal.lfilter_zi(self.snare_b, self.snare_a)
        self.hihat_filter_state = signal.lfilter_zi(self.hihat_b, self.hihat_a)
    
    def setup_filters(self):
        nyquist = self.sample_rate / 2
        
        kick_cutoff = 80 / nyquist
        self.kick_b, self.kick_a = signal.butter(4, kick_cutoff, btype='low')
        
        snare_low = 150 / nyquist
        snare_high = 400 / nyquist
        self.snare_b, self.snare_a = signal.butter(4, [snare_low, snare_high], btype='band')
        
        hihat_cutoff = 8000 / nyquist
        self.hihat_b, self.hihat_a = signal.butter(4, hihat_cutoff, btype='high')
    
    def envelope_follower(self, signal_val, current_env, attack_coeff, release_coeff):
        signal_abs = abs(signal_val)
        if signal_abs > current_env:
            return current_env + attack_coeff * (signal_abs - current_env)
        else:
            return current_env + release_coeff * (signal_abs - current_env)
    
    def zero_crossing_rate(self, audio_chunk):
        if len(self.audio_buffer) < 2:
            return 0
        
        buffer_array = np.array(self.audio_buffer)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(buffer_array))))
        return zero_crossings / (2 * len(buffer_array))
    
    def spectral_centroid(self, audio_chunk):
        fft = np.fft.rfft(audio_chunk)
        magnitude = np.abs(fft)
        
        freqs = np.fft.rfftfreq(len(audio_chunk), 1/self.sample_rate)
        
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            return np.clip(centroid, 20, 20000)
        return 1000
    
    def process_audio_chunk(self, audio_chunk):
        current_time = time.time() - self.start_time
        
        kick_filtered, self.kick_filter_state = signal.lfilter(
            self.kick_b, self.kick_a, audio_chunk, zi=self.kick_filter_state)
        snare_filtered, self.snare_filter_state = signal.lfilter(
            self.snare_b, self.snare_a, audio_chunk, zi=self.snare_filter_state)
        hihat_filtered, self.hihat_filter_state = signal.lfilter(
            self.hihat_b, self.hihat_a, audio_chunk, zi=self.hihat_filter_state)
        
        self.audio_buffer.extend(audio_chunk)
        
        for i, sample in enumerate(audio_chunk):
            self.sample_count += 1
            
            self.kick_env = self.envelope_follower(
                kick_filtered[i], self.kick_env, self.kick_attack, self.kick_release)
            self.snare_env = self.envelope_follower(
                snare_filtered[i], self.snare_env, self.snare_attack, self.snare_release)
            self.hihat_env = self.envelope_follower(
                hihat_filtered[i], self.hihat_env, self.hihat_attack, self.hihat_release)
            self.total_env = self.envelope_follower(
                sample, self.total_env, self.total_attack, self.total_release)
            
            self.check_triggers(current_time + (i / self.sample_rate), audio_chunk)
    
    def check_triggers(self, timestamp, audio_chunk):
        
        kick_condition = (
            self.kick_env > (self.threshold * 2) and
            self.kick_env > (self.snare_env + 0.01) and
            self.kick_env > (self.hihat_env * 2) and
            (self.sample_count - self.last_kick_trigger) > self.kick_holdoff
        )
        
        if kick_condition:
            self.last_kick_trigger = self.sample_count
            self.on_kick_detected(timestamp, audio_chunk)
        
        snare_condition = (
            self.snare_env > self.threshold and
            self.total_env > (self.threshold * 0.5) and
            self.snare_env > (self.kick_env * 0.4) and
            (self.sample_count - self.last_snare_trigger) > self.snare_holdoff
        )
        
        if snare_condition:
            self.last_snare_trigger = self.sample_count
            self.on_snare_detected(timestamp, audio_chunk)
        
        zcr = self.zero_crossing_rate(audio_chunk)
        hihat_condition = (
            self.hihat_env > (self.threshold * 0.3) and
            self.hihat_env > (self.kick_env * 1.5) and
            zcr > 0.1 and
            (self.sample_count - self.last_hihat_trigger) > self.hihat_holdoff
        )
        
        if hihat_condition:
            self.last_hihat_trigger = self.sample_count
            self.on_hihat_detected(timestamp, audio_chunk)
    
    def on_kick_detected(self, timestamp, audio_chunk):
        centroid = self.spectral_centroid(audio_chunk)
        zcr = self.zero_crossing_rate(audio_chunk)
        kick_dominance = self.kick_env / (self.snare_env + 0.001)
        kick_vs_hihat = self.kick_env / (self.hihat_env + 0.001)
        
        print(f"KICK @ {timestamp:.2f}s - Level: {self.kick_env:.3f}, "
              f"Centroid: {centroid:.0f}Hz, Dominance: {kick_dominance:.1f}")
    
    def on_snare_detected(self, timestamp, audio_chunk):
        centroid = self.spectral_centroid(audio_chunk)
        zcr = self.zero_crossing_rate(audio_chunk)
        snare_vs_kick = self.snare_env / (self.kick_env + 0.001)
        snare_vs_hihat = self.snare_env / (self.hihat_env + 0.001)
        
        print(f"SNARE @ {timestamp:.2f}s - Level: {self.snare_env:.3f}, "
              f"Centroid: {centroid:.0f}Hz, ZCR: {zcr:.2f}")
    
    def on_hihat_detected(self, timestamp, audio_chunk):
        centroid = self.spectral_centroid(audio_chunk)
        zcr = self.zero_crossing_rate(audio_chunk)
        hihat_vs_kick = self.hihat_env / (self.kick_env + 0.001)
        hihat_vs_snare = self.hihat_env / (self.snare_env + 0.001)
        
        print(f"HI-HAT @ {timestamp:.2f}s - Level: {self.hihat_env:.3f}, "
              f"Centroid: {centroid:.0f}Hz, ZCR: {zcr:.2f}")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
        
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        if len(audio_data) > 0:
            self.process_audio_chunk(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def start(self):
        if self.running:
            return
        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            self.running = True
            self.start_time = time.time()
            self.sample_count = 0
            
            print(f"detector started (threshold: {self.threshold})")
            print("analyzing...")
            
            self.stream.start_stream()
            
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping detector...")
                self.stop()
                
        except Exception as e:
            print(f"Error starting detector: {e}")
            self.stop()
    
    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.audio.terminate()
        print("Drum detector stopped")
    
    def set_threshold(self, threshold):
        self.threshold = threshold
        print(f"Threshold set to: {threshold}")
    
    def list_audio_devices(self):
        print("Available audio input devices:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"  {i}: {device_info['name']} "
                      f"(channels: {device_info['maxInputChannels']}, "
                      f"rate: {device_info['defaultSampleRate']})")


if __name__ == "__main__":
    detector = DrumDetector(threshold=0.23)
    detector.list_audio_devices()
    
    try:
        detector.start()
    except KeyboardInterrupt:
        print("exit")
    finally:
        detector.stop()
