from bs4 import BeautifulSoup
import requests
import json

url = "https://arxiv.org/category_taxonomy"
page = requests.get(url).content

soup = BeautifulSoup(page, 'html.parser')
h4 = soup.find_all('h4')

category = [h4_i.text for h4_i in h4]
category = category[1:]

mapping = {}

for cat in category:
    sep = cat.find(" (")
    abbrv = cat[:sep]
    full_name = cat[sep+2:]
    full_name = full_name[:-1]
    mapping[abbrv] = full_name


json.dump(mapping, open("data/mapping.json","w"))