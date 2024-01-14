from lib.createPersonData import createPersonData
from lib.audio_pessoa_desconhecida import play_welcome,name_prompt
from lib.audio_pessoa_conhecida import hello_again
from lib.computeIOU import computeIOU
play_welcome()
person_name = name_prompt()
goodbye(person_name)
hello_again(person_name)

