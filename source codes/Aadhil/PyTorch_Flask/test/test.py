import requests



resp = requests.post('http://localhost:5000/predict' , files={'file':open('five.png','rb')})


print(resp.text) 