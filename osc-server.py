#!/usr/bin/env python3
"""
Simple OSC Test Server
Receives and displays OSC messages from butcher
"""
from pythonosc import dispatcher
from pythonosc import osc_server
import sys

# Default port
PORT = 9000
BAR = 80

def kick_handler(unused_addr, word, velocity):
    print(f"KICK   | Word: {word:15s} | Velocity: {velocity:.3f} | {'█' * int(velocity * BAR)}")

def snare_handler(unused_addr, word, velocity):
    print(f"SNARE  | Word: {word:15s} | Velocity: {velocity:.3f} | {'█' * int(velocity * BAR)}")

def hihat_handler(unused_addr, word, velocity):
    print(f"HI-HAT | Word: {word:15s} | Velocity: {velocity:.3f} | {'█' * int(velocity * BAR)}")

def default_handler(address, *args):
    print(f"{address}: {args}")

def main():
    global PORT
    print("\n" + "="*60)
    print("OSC TEST SERVER")
    print("="*60)

    if len(sys.argv) > 1:
        PORT = int(sys.argv[1])

    print(f"OSC server listening on 127.0.0.1:{PORT}")
    print("Listening for butcher")

    disp = dispatcher.Dispatcher()
    disp.map("/butcher/kick", kick_handler)
    disp.map("/butcher/snare", snare_handler)
    disp.map("/butcher/hihat", hihat_handler)
    disp.set_default_handler(default_handler)

    server = osc_server.ThreadingOSCUDPServer(
        ("127.0.0.1", PORT), disp)

    print("="*60)
    print("beep boop...\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nStopping server...")
    finally:
        server.shutdown()
        print("Server closed.")

if __name__ == "__main__":
    main()
