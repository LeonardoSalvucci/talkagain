from email.mime import audio
from gtts import gTTS
from tempfile import NamedTemporaryFile
from playsound import playsound

def speach(txt="Hola Mundo!"):
  voice = NamedTemporaryFile()
  gTTS(txt, lang="es", slow=False).write_to_fp(voice)
  playsound(voice.name)
  voice.close()