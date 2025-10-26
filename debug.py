#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import sys

BUFFER_SIZE = 2048
selected_device = None
sample_rate = None
selected_channel = None


def list_audio_devices():
    print("\n" + "="*60)
    print("AVAILABLE AUDIO INPUT DEVICES")
    print("="*60)
    
    devices = sd.query_devices()
    input_devices = []
    
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append(idx)
            print(f"[{idx}] {device['name']}")
            print(f"    Channels: {device['max_input_channels']}, "
                  f"Sample Rate: {int(device['default_samplerate'])} Hz")
            print()
    
    return input_devices


def select_device():
    input_devices = list_audio_devices()
    
    if not input_devices:
        print("ERROR: No input devices found")
        sys.exit(1)
    
    while True:
        try:
            choice = input("Select device index: ")
            device_idx = int(choice)
            
            if device_idx in input_devices:
                device_info = sd.query_devices(device_idx)
                print(f"\nSelected: {device_info['name']}")
                print(f"Sample Rate: {int(device_info['default_samplerate'])} Hz")
                print(f"Available Channels: {device_info['max_input_channels']}")
                break
            else:
                print(f"Invalid device index. choose from: {input_devices}")
        except ValueError:
            print("enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting")
            sys.exit(0)
    
    max_channels = device_info['max_input_channels']
    selected_channel = None
    
    if max_channels > 1:
        print(f"\nThis device has {max_channels} input channels.")
        print("Which channel/input is your audio source connected to?")
        
        for i in range(max_channels):
            print(f"  [{i+1}] Input {i+1} (Channel {i+1})")
        
        while True:
            try:
                channel_choice = input(f"\nSelect input (1-{max_channels}): ")
                channel_num = int(channel_choice)
                
                if 1 <= channel_num <= max_channels:
                    selected_channel = channel_num - 1
                    print(f"Using Input {channel_num}\n")
                    break
                else:
                    print(f"enter a number between 1 and {max_channels}")
            except ValueError:
                print("enter a valid number")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
    else:
        selected_channel = 0
        print("Using single available input\n")
    
    return device_idx, int(device_info['default_samplerate']), selected_channel


def find_dominant_frequency(audio_data, sample_rate):
    fft_data = np.fft.rfft(audio_data)
    
    magnitude = np.abs(fft_data)
    
    peak_idx = np.argmax(magnitude)
    
    frequency = peak_idx * sample_rate / len(audio_data)
    
    return frequency, magnitude[peak_idx]


def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    
    if indata.shape[1] > 1:
        audio_data = indata[:, selected_channel]  
    else:
        audio_data = indata[:, 0]
    
    audio_level = np.abs(audio_data).max()
    
    frequency, magnitude = find_dominant_frequency(audio_data, sample_rate)
    
    if audio_level > 0.001:  
        print(f"Freq: {frequency:7.2f} Hz  |  Mag: {magnitude:10.2f}  |  Level: {audio_level:.4f}")


def main():
    global selected_device, sample_rate, selected_channel
    
    print("\n" + "="*60)
    print("REAL-TIME FREQUENCY DETECTOR")
    print("="*60)
    print("This script will analyze audio and display the dominant frequency")
    print("Press Ctrl+C to stop")
    
    selected_device, sample_rate, selected_channel = select_device()
    
    print("Starting audio stream...")
    print("Listening...\n")
    
    try:
        with sd.InputStream(
            device=selected_device,
            channels=2,
            samplerate=sample_rate,
            blocksize=BUFFER_SIZE,
            callback=audio_callback
        ):
            print("Press Ctrl+C to stop\n")
            while True:
                sd.sleep(1000)
                
    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("Audio stream closed.")


if __name__ == "__main__":
    main()
