import requests
import json

r1 = requests.get('https://api.semanticscholar.org/datasets/v1/release').json()
print(r1[-3:])

r2 = requests.get('https://api.semanticscholar.org/datasets/v1/release/latest').json()
print(r2['release_id'])


print(json.dumps(r2['datasets'], indent=2))


r3 = requests.get('https://api.semanticscholar.org/datasets/v1/release/latest/dataset/papers').json()
print(json.dumps(r3, indent=2))
