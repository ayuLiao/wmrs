import gzip
import json
import pymysql

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))


db = pymysql.connect('localhost', 'root', 'AyuLiao*666', 'wmrsdb')
cursor = db.cursor()

path = '/Users/ayuliao/Desktop/workplace/word2vec-recommender/data/metadata.json.gz'
i = 0
for l in parse(path):
    l = eval(l)
    sql = 'INSERT INTO product_productinfo(asin,imUrl, price, title) VALUES ("%s", "%s", "%f", "%s")' % \
          (l.get('asin',''), l.get('imUrl',''), l.get('price',0), l.get('title',''))
    try:
        cursor.execute(sql)
        db.commit()
    except:
        db.rollback()
    print(i)
    i = i+1

db.close()