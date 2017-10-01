import json
import collections

with open('../train/train.json', 'r') as f:
    jd = json.load(f)

print(len(jd))

line = jd[0]
print(jd[0])

c = collections.Counter()

for data in jd:
    c.update([data['label_id']])
print(c)

total = 0
for k, v in c.items():
    total += v

print(total)