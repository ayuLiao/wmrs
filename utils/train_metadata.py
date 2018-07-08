# -*- coding: utf-8 -*-
import gensim
import gzip
import json
import os

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

class MySentences(object):
    def __init__(self,path):
        # metadata元数据的路径
        self.path = path

    def __iter__(self):
        i = 0
        for l in parse(self.path):
            l = eval(l)
            asin = l.get('asin', '')
            related = l.get('related', {})
            if related:
                also_bought = related.get('also_bought', [])
                also_viewed = related.get('also_viewed', [])
                bought_together = related.get('bought_together', [])
                buy_after_viewing = related.get('buy_after_viewing', [])
                sents = asin + ' ' + ' '.join(also_bought) + ' ' + ' '.join(also_viewed) + ' ' + ' '.join(bought_together)+ ' ' + ' '.join(buy_after_viewing)
                yield sents.split()
            else:
                continue
            print(i)
            i +=1




def train(path, model_path):
    sentences = MySentences(path)
    '''
    `min_count` =忽略总频率低于此值的所有单词。
    `sg`定义了训练算法。 默认情况下（`sg = 0`），使用CBOW。否则（`sg = 1`），使用skip-gram。
    `negative': if> 0，将使用负采样，int为负数指定应绘制多少“噪音词”（通常在5-20之间）。默认值是5.如果设置为0，则不使用负面抽样。
    `sample` :配置哪些较高频率的字随机下采样的阈值;默认值是1e-3，有用的范围是（0,1e-5）。
    `hs` = if 1，等级softmax将用于模型训练。如果设置为0（默认），并且'negative'不为零，则将使用负采样。
    '''
    for name in os.listdir(path):
        filepath = path+name
        sentences = MySentences(filepath)
        gensim.models.Word2Vec.load
        model = gensim.models.Word2Vec(sentences, min_count=10, workers=4, negative=10, size=300, sample=1e-4, sg=0,
                                       hs=1, window=10)
        model.init_sims(replace=True)
        model.save(model_path+'_'+name.split('.')[0])


if __name__ == '__main__':
    path = '/Users/ayuliao/Desktop/workplace/wmrs/data2/'
    model_path = '/Users/ayuliao/Desktop/workplace/wmrs/models/Model'
    train(path, model_path)
