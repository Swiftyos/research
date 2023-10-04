import cv2
import sounddevice as sd
import wave
import threading
import time
import numpy as np

# Define the duration of the video in seconds
duration = 5

# Open the webcam
cap = cv2.VideoCapture(0)

# Define the codec using VideoWriter_fourcc() and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

# Audio recording parameters
chunk = 1024  # Record in chunks of 1024 samples
channels = 2
fs = 44100  # Record at 44100 samples per second

# Start the audio stream
stream = sd.InputStream(samplerate=fs, channels=channels, blocksize=chunk)
stream.start()
frames = []

# Get the starting time
start_time = time.time()


def record_audio(stream, frames):
    while True:
        data = stream.read(chunk)
        frames.append(data)
        if time.time() - start_time > duration:
            break


# Start the audio recording thread
audio_thread = threading.Thread(target=record_audio, args=(stream, frames))
audio_thread.start()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Write the frame
        out.write(frame)

        # Display the resulting frame
        cv2.imshow("frame", frame)

    # Break the loop when the duration is reached
    if time.time() - start_time > duration:
        break

# Release everything after the capture
cap.release()
out.release()
cv2.destroyAllWindows()

# Stop and close the stream
stream.stop()

# Save the audio to a .wav file
wf = wave.open("output.wav", "wb")
wf.setnchannels(channels)
wf.setsampwidth(np.dtype(np.int16).itemsize)
wf.setframerate(fs)
wf.writeframes(b"".join(frames))
wf.close()
