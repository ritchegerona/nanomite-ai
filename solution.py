import requests

url = "http://example.com/login"
params = {"username": "admin", "password": "123456"}

response = requests.post(url, params=params)
print(response.text)