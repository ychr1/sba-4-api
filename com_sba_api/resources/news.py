from com_sba_api.ext.db import db 
from typing import List
from flask import request
from flask_restful import Resource, reqparse

import json
from flask import jsonify



# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================


# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================

'''
class News_(Resource):
    def get(self):
        return {'news': list(map(lambda news: news.json(), NewsDao.find_all()))}
        #return {'kospis':[kospi.json() for kospi in KospiDao.find_all()]}
class NewsDto(db.Model):
    __tablename__ = 'naver_news'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    news_id : int = db.Column(db.String(30), primary_key = True, index=True)
    # date : datetime = db.Column(db.datetime)
    sentiment_analysis :str = db.Column(db.String(30))
    keywords :str = db.Column(db.String(30))
    
    def __init__(self, news_id, date, sentiment_analysis, keywords):
        self.news_id = news_id
        self.date = date
        self.sentiment_analysis = sentiment_analysis
        self.keywords = keywords
        
    
    def __repr__(self):
        return f'news_id={self.news_id}, date={self.date}, sentiment_analysis={self.sentiment_analysis},\
            keywords={self.keywords}'
            
    @property
    def json(self):
        return {
            'news_id': self.news_id,
            'date': self.date,
            'sentiment_analysis' : self.sentiment_analysis,
            'keywords' : self.keywords
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

class NewsDao(db.Model):    

    @classmethod
    def find_all(cls):
        return cls.query


    @classmethod
    def find_by_name(cls,name):
        return cls.query.filter_by(name == name)


    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(id==id).first()


class News(Resource):
    
    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('news_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('date', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('symbol', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('headline', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('url', type=str, required=False, help='This field cannot be left blank')

    def post(self):
        data = self.parset.parse_args()
        news = NewsDto(data['date'],data['symbol'],data['headline'],data['url'])
        try:
            news.save()
        except:
            return {'message':'An error occured inserting the news'}, 500
        return news.json(), 201

    def get(self,news_id):
        news = NewsDao.find_by_id(news_id)
        if news:
            return news.json()
        return {'message': 'News not found'}, 404

    def put (self, news_id):
        data = News.parser.parse_args()
        news = NewsDto.find_by_id(news_id)

        news.date = data['date']
        news.stock = data['symbol']
        news.price= data['headline']
        news.price= data['url']
        news.save()
        return news.json()

'''