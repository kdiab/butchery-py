import numpy as np
import pyaudio
import time
from scipy import signal
from collections import deque


class DrumDetector:
    def __init__(self, 
                 sample_rate=48000, 
                 chunk_size=1024, 
                 threshold=0.23,
                 input_device=2,
                 # Kick detection parameters
                 kick_threshold_mult=3.5,
                 kick_snare_ratio=2.5,
                 kick_hihat_ratio=3.0,
                 kick_energy_mult=1.5,
                 # Snare detection parameters
                 snare_threshold_mult=1.2,
                 snare_energy_mult=0.7,
                 snare_kick_ratio=0.5,
                 # Hi-hat detection parameters
                 hihat_threshold_mult=0.15,
                 hihat_kick_ratio=0.8,
                 hihat_zcr_min=0.05,
                 # Debug mode
                 debug=False):
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.input_device = input_device
        self.debug = debug
        
        self.kick_threshold_mult = kick_threshold_mult
        self.kick_snare_ratio = kick_snare_ratio
        self.kick_hihat_ratio = kick_hihat_ratio
        self.kick_energy_mult = kick_energy_mult
        
        self.snare_threshold_mult = snare_threshold_mult
        self.snare_energy_mult = snare_energy_mult
        self.snare_kick_ratio = snare_kick_ratio
        
        self.hihat_threshold_mult = hihat_threshold_mult
        self.hihat_kick_ratio = hihat_kick_ratio
        self.hihat_zcr_min = hihat_zcr_min
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        
        self.start_time = time.time()
        self.sample_count = 0
        
        self.debug_counter = 0
        self.debug_interval = int(sample_rate * 0.5)  # Print debug every 0.5 seconds
        
        self.kick_holdoff = int(0.15 * sample_rate)   # 150ms
        self.snare_holdoff = int(0.1 * sample_rate)   # 100ms
        self.hihat_holdoff = int(0.06 * sample_rate)  # 60ms
        
        self.last_kick = 0
        self.last_snare = 0
        self.last_hihat = 0
        
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
        
        self.zcr_buffer_size = int(0.02 * sample_rate)
        self.audio_buffer = deque(maxlen=self.zcr_buffer_size)
        
        self.design_filters()
        
        self.filters_initialized = False
        self.kick_zi = None
        self.snare_zi = None
        self.hihat_zi = None
    
    def design_filters(self):
        """Design Butterworth filters for each drum type"""
        nyquist = self.sample_rate / 2
        
        kick_freq = 80 / nyquist
        self.kick_b, self.kick_a = signal.butter(4, kick_freq, btype='low')
        
        snare_low = 150 / nyquist
        snare_high = 400 / nyquist
        self.snare_b, self.snare_a = signal.butter(4, [snare_low, snare_high], btype='band')
        
        hihat_freq = 8000 / nyquist
        self.hihat_b, self.hihat_a = signal.butter(4, hihat_freq, btype='high')
    
    def init_filter_states(self, first_sample):
        """Initialize filter states with first sample to avoid transients"""
        self.kick_zi = signal.lfilter_zi(self.kick_b, self.kick_a) * first_sample
        self.snare_zi = signal.lfilter_zi(self.snare_b, self.snare_a) * first_sample
        self.hihat_zi = signal.lfilter_zi(self.hihat_b, self.hihat_a) * first_sample
        self.filters_initialized = True
    
    def envelope_follower(self, sample, current_env, attack, release):
        """Track amplitude envelope of a signal"""
        sample_abs = abs(sample)
        if sample_abs > current_env:
            return current_env + attack * (sample_abs - current_env)
        else:
            return current_env + release * (sample_abs - current_env)
    
    def calculate_zcr(self):
        """Calculate zero crossing rate"""
        if len(self.audio_buffer) < 2:
            return 0.0
        
        buffer = np.array(self.audio_buffer)
        crossings = np.sum(np.abs(np.diff(np.sign(buffer))))
        return crossings / (2 * len(buffer))
    
    def calculate_spectral_centroid(self, chunk):
        """Calculate spectral centroid (brightness)"""
        fft = np.fft.rfft(chunk)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(chunk), 1 / self.sample_rate)
        
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            return np.clip(centroid, 20, 20000)
        return 1000.0
    
    def process_chunk(self, audio_chunk):
        """Process a chunk of audio"""
        current_time = time.time() - self.start_time
        
        if not self.filters_initialized:
            self.init_filter_states(audio_chunk[0])
        
        kick_band, self.kick_zi = signal.lfilter(
            self.kick_b, self.kick_a, audio_chunk, zi=self.kick_zi)
        
        snare_band, self.snare_zi = signal.lfilter(
            self.snare_b, self.snare_a, audio_chunk, zi=self.snare_zi)
        
        hihat_band, self.hihat_zi = signal.lfilter(
            self.hihat_b, self.hihat_a, audio_chunk, zi=self.hihat_zi)
        
        self.audio_buffer.extend(audio_chunk)
        
        for i in range(len(audio_chunk)):
            self.sample_count += 1
            
            self.kick_env = self.envelope_follower(
                kick_band[i], self.kick_env, self.kick_attack, self.kick_release)
            
            self.snare_env = self.envelope_follower(
                snare_band[i], self.snare_env, self.snare_attack, self.snare_release)
            
            self.hihat_env = self.envelope_follower(
                hihat_band[i], self.hihat_env, self.hihat_attack, self.hihat_release)
            
            self.total_env = self.envelope_follower(
                audio_chunk[i], self.total_env, self.total_attack, self.total_release)
            
            self.detect_drums(current_time + (i / self.sample_rate), audio_chunk)
    
    def detect_drums(self, timestamp, chunk):
        """Check if any drums should trigger"""
        
        if self.debug:
            self.debug_counter += 1
            if self.debug_counter >= self.debug_interval:
                self.debug_counter = 0
                print(f"\n[DEBUG @ {timestamp:.2f}s]")
                print(f"  Kick env:  {self.kick_env:.4f} (needs > {self.threshold * self.kick_threshold_mult:.4f})")
                print(f"  Snare env: {self.snare_env:.4f} (needs > {self.threshold * self.snare_threshold_mult:.4f})")
                print(f"  Hihat env: {self.hihat_env:.4f} (needs > {self.threshold * self.hihat_threshold_mult:.4f})")
                print(f"  Total env: {self.total_env:.4f}")
                print(f"  ZCR: {self.calculate_zcr():.4f}")
                
                print(f"\n  Kick checks:")
                print(f"    Env > threshold*mult: {self.kick_env:.4f} > {self.threshold * self.kick_threshold_mult:.4f} = {self.kick_env > (self.threshold * self.kick_threshold_mult)}")
                print(f"    Env > snare*ratio: {self.kick_env:.4f} > {self.snare_env * self.kick_snare_ratio:.4f} = {self.kick_env > (self.snare_env * self.kick_snare_ratio)}")
                print(f"    Env > hihat*ratio: {self.kick_env:.4f} > {self.hihat_env * self.kick_hihat_ratio:.4f} = {self.kick_env > (self.hihat_env * self.kick_hihat_ratio)}")
                print(f"    Total > threshold*mult: {self.total_env:.4f} > {self.threshold * self.kick_energy_mult:.4f} = {self.total_env > (self.threshold * self.kick_energy_mult)}")
        
        kick_ok = (
            self.kick_env > (self.threshold * self.kick_threshold_mult) and
            self.kick_env > (self.snare_env * self.kick_snare_ratio) and
            self.kick_env > (self.hihat_env * self.kick_hihat_ratio) and
            self.total_env > (self.threshold * self.kick_energy_mult) and
            (self.sample_count - self.last_kick) > self.kick_holdoff
        )
        
        if kick_ok:
            self.last_kick = self.sample_count
            self.trigger_kick(timestamp, chunk)
        
        snare_ok = (
            self.snare_env > (self.threshold * self.snare_threshold_mult) and
            self.total_env > (self.threshold * self.snare_energy_mult) and
            self.snare_env > (self.kick_env * self.snare_kick_ratio) and
            (self.sample_count - self.last_snare) > self.snare_holdoff
        )
        
        if snare_ok:
            self.last_snare = self.sample_count
            self.trigger_snare(timestamp, chunk)
        
        zcr = self.calculate_zcr()
        hihat_ok = (
            self.hihat_env > (self.threshold * self.hihat_threshold_mult) and
            self.hihat_env > (self.kick_env * self.hihat_kick_ratio) and
            zcr > self.hihat_zcr_min and
            (self.sample_count - self.last_hihat) > self.hihat_holdoff
        )
        
        if hihat_ok:
            self.last_hihat = self.sample_count
            self.trigger_hihat(timestamp, chunk)
    
    def trigger_kick(self, timestamp, chunk):
        """Handle kick drum detection"""
        centroid = self.calculate_spectral_centroid(chunk)
        dominance = self.kick_env / (self.snare_env + 0.001)
        
        print(f"KICK @ {timestamp:.2f}s - Level: {self.kick_env:.3f}, "
              f"Centroid: {centroid:.0f}Hz, Dominance: {dominance:.1f}")
    
    def trigger_snare(self, timestamp, chunk):
        """Handle snare drum detection"""
        centroid = self.calculate_spectral_centroid(chunk)
        zcr = self.calculate_zcr()
        
        print(f"SNARE @ {timestamp:.2f}s - Level: {self.snare_env:.3f}, "
              f"Centroid: {centroid:.0f}Hz, ZCR: {zcr:.2f}")
    
    def trigger_hihat(self, timestamp, chunk):
        """Handle hi-hat detection"""
        centroid = self.calculate_spectral_centroid(chunk)
        zcr = self.calculate_zcr()
        
        print(f"HI-HAT @ {timestamp:.2f}s - Level: {self.hihat_env:.3f}, "
              f"Centroid: {centroid:.0f}Hz, ZCR: {zcr:.2f}")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Stream status: {status}")
        
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        if len(audio_data) > 0:
            self.process_chunk(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def start(self):
        """Start the drum detector"""
        if self.running:
            print("Detector already running")
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
            
            print(f"\nDrum Detector Started")
            print(f"Sample Rate: {self.sample_rate} Hz")
            print(f"Threshold: {self.threshold}")
            print(f"Input Device: {self.input_device}")
            print("\nListening for drums... Press Ctrl+C to stop\n")
            
            self.stream.start_stream()
            
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\nStopping detector...")
                self.stop()
        
        except Exception as e:
            print(f"Error starting detector: {e}")
            self.stop()
    
    def stop(self):
        """Stop the drum detector"""
        self.running = False
        
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.audio.terminate()
        print("Detector stopped\n")
    
    def set_threshold(self, new_threshold):
        """Change detection threshold"""
        self.threshold = new_threshold
        print(f"Threshold updated to: {new_threshold}")
    
    def set_kick_params(self, threshold_mult=None, snare_ratio=None, hihat_ratio=None, energy_mult=None):
        """Fine-tune kick detection parameters"""
        if threshold_mult is not None:
            self.kick_threshold_mult = threshold_mult
        if snare_ratio is not None:
            self.kick_snare_ratio = snare_ratio
        if hihat_ratio is not None:
            self.kick_hihat_ratio = hihat_ratio
        if energy_mult is not None:
            self.kick_energy_mult = energy_mult
        
        print(f"Kick params: thresh_mult={self.kick_threshold_mult}, "
              f"snare_ratio={self.kick_snare_ratio}, hihat_ratio={self.kick_hihat_ratio}, "
              f"energy_mult={self.kick_energy_mult}")
    
    def set_snare_params(self, threshold_mult=None, energy_mult=None, kick_ratio=None):
        """Fine-tune snare detection parameters"""
        if threshold_mult is not None:
            self.snare_threshold_mult = threshold_mult
        if energy_mult is not None:
            self.snare_energy_mult = energy_mult
        if kick_ratio is not None:
            self.snare_kick_ratio = kick_ratio
        
        print(f"Snare params: thresh_mult={self.snare_threshold_mult}, "
              f"energy_mult={self.snare_energy_mult}, kick_ratio={self.snare_kick_ratio}")
    
    def set_hihat_params(self, threshold_mult=None, kick_ratio=None, zcr_min=None):
        """Fine-tune hi-hat detection parameters"""
        if threshold_mult is not None:
            self.hihat_threshold_mult = threshold_mult
        if kick_ratio is not None:
            self.hihat_kick_ratio = kick_ratio
        if zcr_min is not None:
            self.hihat_zcr_min = zcr_min
        
        print(f"Hi-hat params: thresh_mult={self.hihat_threshold_mult}, "
              f"kick_ratio={self.hihat_kick_ratio}, zcr_min={self.hihat_zcr_min}")
    
    def list_devices(self):
        """List all available audio input devices"""
        print("\nAvailable Audio Input Devices:")
        print("-" * 60)
        
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
                print(f"      Channels: {info['maxInputChannels']}, "
                      f"Sample Rate: {info['defaultSampleRate']:.0f} Hz")
        
        print("-" * 60 + "\n")


if __name__ == "__main__":
    detector = DrumDetector(
        sample_rate=48000,
        threshold=0.04,
        input_device=2,  # BlackHole 2ch
        
        kick_threshold_mult=2.0, 
        kick_snare_ratio=2.0,     
        kick_hihat_ratio=1.5,       
        kick_energy_mult=1.2,       
        
        snare_threshold_mult=0.5,   
        snare_energy_mult=0.4,     
        snare_kick_ratio=0.3,     
        
        hihat_threshold_mult=0.15,
        hihat_kick_ratio=0.8,
        hihat_zcr_min=0.05,
        
    )
    
    # Show available devices
    detector.list_devices()
    
#    detector.set_kick_params(threshold_mult=1.5)  # Make kick MORE sensitive
    detector.set_hihat_params(threshold_mult=0.1)  # Make hi-hat more sensitive
  #  detector.set_snare_params(threshold_mult=1.5)  # Make snare less sensitive
    
    # Start detecting
    try:
        detector.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        detector.stop()
