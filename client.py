import requests
import json
import cv2
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig

addr = 'http://127.0.0.1:5000'
test_url = addr + '/predict_api'
content_type = 'image/jpeg'
headers = {'content-type': content_type}
print("read img")
img = cv2.imread('images/t21.jpg')
_, img_encoded = cv2.imencode('.jpg', img)
print("send img")
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
print("recv img")
pred = json.loads(response.text)
query = pred["pred"]
stopwords = ['startseq','endseq']
querywords = query.split()
resultwords  = [word for word in querywords if word.lower() not in stopwords]
result = ' '.join(resultwords)
print(result)
res = '<speak version="1.0" xmlns="https://www.w3.org/2001/10/synthesis" xml:lang="en-US"><voice name="en-US-Guy24kRUS">'+result+'</voice></speak>'
subscription_key = '639cbe821c074e68ba19be3d46a9cbda'
speech_config = SpeechConfig(subscription=subscription_key, region="centralindia")
audio_config = AudioOutputConfig(use_default_speaker=True)
synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
synthesizer.speak_ssml_async(res)



