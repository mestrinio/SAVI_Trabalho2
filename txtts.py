import pygame
import gtts
from playsound import playsound

def text_to_speech(text, language='en', speed='normal'):

  # Create a gTTS object.
  tts = gtts.gTTS(text, lang=language, tld='ca', slow=False if speed == 'normal' else True)

  # Save the audio file.
  audio_file = 'audio.mp3'
  tts.save(audio_file)

  return audio_file


def txt_speech():
  print('ESTA AQUI O DEBUG')
  playsound('audio.mp3')


    

