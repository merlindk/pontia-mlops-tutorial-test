import requests

API_URL = "http://model-api-test-15761373539.eastus.azurecontainer.io:8080/predict"

payload = {
  "age": 38,
  "workclass": "Private",
  "fnlwgt": 89814,
  "education": "HS-grad",
  "education-num": 9,
  "marital-status": "Married-civ-spouse",
  "occupation": "Farming-fishing",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 50,
  "native-country": "United-States"
}

response = requests.post(API_URL, json=payload)

if response.ok:
    print("Prediction:", response.json()["prediction"])
else:
    print("Error:", response.status_code, response.text)
