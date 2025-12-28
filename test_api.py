import requests

url = "http://127.0.0.1:5002/run"

payload = {
    
    "username": "test_user"
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())
