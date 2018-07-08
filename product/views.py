from django.shortcuts import render
from django.views.generic.base import View
from .models import ProductInfo

from gensim.models import word2vec

import os
import random


model_path = '/Users/ayuliao/Desktop/workplace/wmrs/models/'
models = []
for m in os.listdir(model_path):
    if '.' not in m:
        models.append(word2vec.Word2Vec.load(model_path+m))

# Create your views here.
class IndexListView(View):
    def get(self, request):
        five_product = []
        for i in range(5):
            five_product.append(ProductInfo.objects.get(id=random.randint(5, 9352950)))
        product_dict = {}
        for product in five_product:
            try:
                product.imUrl = 'https://images-na.ssl-images-amazon.com/images/'+product.imUrl.split('/images/')[1]
                product_dict[product.asin] = [product.imUrl, product.title]
            except:
                pass

        return render(request, 'index.html', context={
            'product_dict':product_dict,
        })


class Word2vecListView(View):
    def get(self, request, asin):
        product_dict = {}
        for model in models:
            try:
                products = model.most_similar(asin)
                for p in products:
                    # p[0]为商品的asin，p[1]为相似度
                    try:
                        product = ProductInfo.objects.get(asin=p[0])
                        product.imUrl = 'https://images-na.ssl-images-amazon.com/images/'+product.imUrl.split('/images/')[1]
                        product_dict[product.asin] = [product.imUrl, product.title, p[1]]
                    except:
                        pass
            except: #在model中没有要找的词则会报错
                pass
        if product_dict:
            return render(request, 'like.html', context={
                'product_dict': product_dict,
            })
        else:
            return render(request, 'error.html', context={
                'error': '该商品没有相似推荐',
            })

