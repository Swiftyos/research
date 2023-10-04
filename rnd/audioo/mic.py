import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


def list_devices():
    """List all available audio devices."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(
            f"{i}: {device['name']} (Channels: {device['max_input_channels']}/{device['max_output_channels']})"
        )


def record_audio(device_index, duration, samplerate=44100):
    """Record audio from a selected device."""
    device_info = sd.query_devices(device_index)
    channels = device_info["max_input_channels"]

    print(f"Recording for {duration} seconds...")
    recording = sd.rec(
        int(samplerate * duration),
        samplerate=samplerate,
        channels=channels,
        device=device_index,
        dtype=np.int16,
    )
    sd.wait()
    return recording


def playback_audio(audio, samplerate=44100):
    """Play back the recorded audio."""
    print("Playing back recorded audio...")
    sd.play(audio, samplerate=samplerate)
    sd.wait()


def main():
    # List devices
    list_devices()

    # Select device
    device_index = int(input("Enter the index of the device you want to use: "))

    # Record audio
    duration = 5  # seconds
    recording = record_audio(device_index, duration)

    # Playback
    playback_audio(recording)

    # Save to file
    filename = "debug_mic_output.wav"
    write(filename, 44100, recording)
    print(f"Audio saved to {filename}")


if __name__ == "__main__":
    main()
