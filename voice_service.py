import os
import time
import pygame
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play


def play_text_to_speech(text, language="en", slow=False, speed=1.2):
    # Generate speech file
    tts = gTTS(text=text, lang=language, slow=slow)
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)

    # Load audio and change speed
    audio = AudioSegment.from_file(temp_audio_file)
    faster_audio = audio.speedup(playback_speed=speed)

    # Save new fast audio
    fast_audio_file = "fast_audio.mp3"
    faster_audio.export(fast_audio_file, format="mp3")

    # Play audio using pygame
    pygame.mixer.init()
    pygame.mixer.music.load(fast_audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.stop()
    pygame.mixer.quit()

    time.sleep(1)  # Shorter delay
    os.remove(temp_audio_file)
    os.remove(fast_audio_file)
