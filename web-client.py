import requests
import json
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig
import cv2
addr = 'https://missionvision.azurewebsites.net'
test_url = addr + '/predict_api'
content_type = 'image/jpeg'
headers = {'content-type': content_type}
my_img = cv2.imread("images/t15.jpg")
my_img = cv2.cvtColor(my_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("temp1.jpg",my_img)
my_img = {'image': open('temp1.jpg', 'rb')}
response = requests.post(test_url, files=my_img)
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
