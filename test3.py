import requests
res = requests.get('http://localhost:9200')
print(res.content)
from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

import json

r = requests.get('http://localhost:9200')
i = 1



abc = {'nom': 'Chiheb El ouekdi', 'poids': '34', 'hauteur': '1.77', 'jdjdj': 'lll'}
#while r.status_code == 200:
#    r = requests.get('http://swapi.co/api/people/' + str(i))
#    es.index(index='sw', doc_type='people', id=i, body=json.loads(r.content))
#    print (json.loads(r.contentchihebch)
#    i = i + 1

es.index(index='chub', doc_type='people', id=i, body=abc)
es.msearch

import wikipedia
print(wikipedia.summary("google"))
print("-----------------------------------------------------")
print(wikipedia.search("google"))
print("-----------------------------------------------------")
page = wikipedia.page("google")
print(page.url)