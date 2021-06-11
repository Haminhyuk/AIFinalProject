
# 음성으로 들려주는 곳
from gtts import gTTS
import playsound
import os

news = "에이아이 시스템 설계는 재미있는 과목입니다."
news2 = "동해물과 백두산이 마르고 닳도록~"

tts=gTTS(text=news, lang='ko')
tts2=gTTS(text=news2, lang='ko')
tts.save("news_Son.mp3") #mp3 파일로 저장을 하겠따.
tts2.save("news_bgm.mp3")

#block True = 다음 함수를 실행하지말고 다 실행하고 해라
playsound.playsound("news_Son.mp3", False)
playsound.playsound("news_bgm.mp3", True)