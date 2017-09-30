import json

with open('../train/label.json', 'r') as f:
    jd = json.load(f)
print(len(jd))

line = jd[0]
print(jd[0])

