import requests
import json


r = requests.post(
    'https://api.semanticscholar.org/graph/v1/paper/batch',
    params={'fields': 'referenceCount,citationCount,title'},
    json={"title": ["MMLU", "MMMU"]}
)
print(json.dumps(r.json(), indent=2))

