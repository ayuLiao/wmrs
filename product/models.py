from django.db import models

# Create your models here.
class ProductInfo(models.Model):
    '''
    商品信息表
    '''
    asin =  models.CharField(null=True, blank=True, max_length=20, verbose_name='商品ID')
    imUrl = models.CharField(null=True, blank=True, max_length=200, verbose_name='商品图片链接')
    price = models.FloatField(null=True, blank=True, verbose_name='商品价格')
    title = models.CharField(null=True, blank=True, max_length=200, verbose_name='商品标题')
    class Meta:
        verbose_name = '商品信息'
        verbose_name_plural = verbose_name