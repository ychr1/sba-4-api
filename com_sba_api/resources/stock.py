from typing import List
from flask import request
from flask_restful import Resource, reqparse
from flask import jsonify
from com_sba_api.ext.db import db, openSession
from com_sba_api.util.file import FileReader
from sklearn.ensemble import RandomForestClassifier # rforest
from sklearn.tree import DecisionTreeClassifier # dtree
from sklearn.ensemble import RandomForestClassifier # rforest
from sklearn.naive_bayes import GaussianNB # nb
from sklearn.neighbors import KNeighborsClassifier # knn
from sklearn.svm import SVC # svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold  # k value is understood as count
from sklearn.model_selection import cross_val_score
from sqlalchemy import func
from pathlib import Path
from sqlalchemy import and_, or_
from datetime import datetime
from pandas._libs.tslibs.offsets import relativedelta
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from dataclasses import dataclass
from konlpy.tag import Okt
from nltk import word_tokenize, re, FreqDist

import FinanceDataReader as fdr # pip install -U 
import pandas as pd
import json
import numpy as np
import pandas_datareader as pdr
import json
import os
import csv
import requests
import re
import collections
import csv
import json
import pandas as pd


'''
 * @ Module Name : stock.py
 * @ Description : Recommendation for share price transactions
 * @ since 2009.03.03
 * @ version 1.0
 * @ Modification Information
 * @ author 주식거래 AI 추천서비스 개발팀 박정관
 * @ special reference libraries
 *     finance_datareader, konlpy
 * @ 수정일         수정자                   수정내용
 *  -------    --------    ---------------------------
 *  2020.08.01    최윤정          최초 생성
 *  2020.10.29    박정관          모듈 통합 및 개선

''' 

# ==============================================================
# =========================                =====================
# =========================  Data Mining   =====================
# =========================                =====================
# ==============================================================

class StockDm:
    
    def candle_crawling(self, symbol):
        symbol = symbol
        symbol_with_ks = symbol + '.KS'

        end_date = datetime.now()
        start_date = datetime.now()-relativedelta(months=3)
        temp = pdr.get_data_yahoo(symbol_with_ks, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        temp.drop(['Volume', 'Adj Close'], axis=1, inplace=True)

        date_index = temp.index.get_level_values('Date').tolist()

        result = []
        for i in range(len(date_index)):
            result.append({'x': str(date_index[i].strftime('%Y-%m-%d')),
                           'y': list(np.array(temp.iloc[i]).tolist())})
        return result

    # ThreeDays:
    # 통합 워드클라우드를 만들게 될지도 몰라 준비하는 크롤링
    # 3일간의 이슈 검색어를 팔로업한다

    def date(self):
        date = []
        today = datetime.today()
        dt_index = pd.date_range(today, periods=2, freq='-1d')
        dt_list = dt_index.strftime("%Y%m%d").tolist()
        for i in dt_list:
            date.append(i)
        return date

    def news_crawling_1(self, page_number):
        result = []
        date = self.date()
        for regDate in date:
            for i in range(page_number):
                url = "https://finance.naver.com/news/news_list.nhn?mode=LSS3D&section_id=101&section_id2=258&section_id3=402" \
                      "&date={date}&page={page}".format(date=regDate, page=i)
                html = requests.get(url).text
                soup = BeautifulSoup(html, 'html.parser')
                a = soup.find_all('dd', {'class': 'articleSubject'})
                for item in a:
                    link = str('https://finance.naver.com{}') \
                        .format(item.find('a')['href']
                                .replace("§", "&sect"))
                    content = self.get_text(link)
                    news = {content: "content"}
                    result.append(news)
        self.get_csv(result)
        return result

    def get_csv(self, result):
        file = open('../static/data/news_threeDays_crawling.csv', 'w', encoding='utf-8', newline='')
        csvfile = csv.writer(file)
        for row in result:
            csvfile.writerow(row)
        file.close()

    def get_text(self, url):
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        content = ''
        for item in soup.find_all('div', {'id': 'content'}):
            for text in item.find_all(text=True):
                if re.search('▶', text) is not None:
                    break
                content = content + text + "\n\n"
        return content


if __name__ == '__main__':
    stockDf = StockDm()
    crawl = stockDf.news_crawling_1(page_number=100)
    



    # 썸네일 포함한 뉴스 5일치 크롤링
    # csv 파일로 생성하여 DB에 저장한다
    # NewsListCrawler:


    def date(self):
            date = []
            selected = '2020-08-25'
            dt_index = pd.date_range(selected, periods=8, freq='-1d')
            dt_list = dt_index.strftime("%Y%m%d").tolist()
            for i in dt_list:
                print(i)
                date.append(i)
            return date

    def news_crawling_2(self, page_number):
        result = []
        date = self.date()
        for regDate in date:
            for i in range(page_number):
                url = "https://finance.naver.com/news/news_list.nhn?mode=LSS3D&section_id=101&section_id2=258&section_id3=402" \
                      "&date={date}&page={page}".format(date=regDate, page=i)
                html = requests.get(url).text
                soup = BeautifulSoup(html, 'html.parser')
                a = soup.find_all('dd', {'class': 'articleSubject'})
                for item in a:
                    title = item.find('a')['title']
                    link = str('https://finance.naver.com{}') \
                        .format(item.find('a')['href']
                                .replace("§", "&sect"))
                    wdate = self.get_wdate(link)
                    content = self.get_text(link)
                    thumbnail = self.get_thumbnail(link)
                    news = {wdate: "wdate", title: "title", content: "content", link: "link", thumbnail: "thumbnail"}
                    result.append(news)
        self.get_csv(result)
        return result

    def get_csv(self, result):
        file = open('../static/data/final_news_crawling.csv', 'w', encoding='utf-8', newline='')
        csvfile = csv.writer(file)
        for row in result:
            csvfile.writerow(row)
        file.close()

    def get_wdate(self, url):
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        written_date = soup.find_all(class_='article_sponsor')
        for date in written_date:
            wdate = date.find('span').text
            return wdate

    def get_thumbnail(self, url):
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        article_image = soup.find_all(class_='end_photo_org')
        for item in article_image:
            src = item.find('img')['src']
            return src

    def get_text(self, url):
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        content = ''
        for item in soup.find_all('div', {'id': 'content'}):
            for text in item.find_all(text=True):
                if re.search('▶', text) is not None:
                    break
                content = content + text + "\n\n"
        return content


if __name__ == '__main__':
    stockDm = StockDm()
    crawl = stockDm.news_crawling_2(page_number=100)



# Text_mining_create_csv.py 

@dataclass
class Entity:
    context: str = ''
    fname: str = ''
    target: str = ''
    date: str = ''

class Service:
    def __init__(self):
        self.texts = []
        self.tokens = []
        self.noun_tokens = []
        self.okt = Okt()
        self.stopword = []
        self.freqtxt = []
        self.date = []

    def tokenize(self):
        filename = r'../static/data/news_threeDays_crawling.csv'
        with open(filename, 'r', encoding='utf-8') as f:
            self.texts = f.read()
        texts = self.texts.replace('\n', '')
        tokenizer = re.compile(r'[^ㄱ-힣]')
        self.texts = tokenizer.sub(' ', texts)
        self.tokens = word_tokenize(self.texts)
        _arr = []
        for token in self.tokens:
            token_pos = self.okt.pos(token)
            _ = [txt_tag[0] for txt_tag in token_pos if txt_tag[1] == 'Noun']
            if len("".join(_)) > 1:
                _arr.append("".join(_))
        self.noun_tokens = " ".join(_arr)

        filename = r'../static/data/stopwords.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            self.stopword = f.read()
        print(type(self.stopword))
        self.noun_tokens = word_tokenize(self.noun_tokens)
        self.noun_tokens = [text for text in self.noun_tokens
                            if text not in self.stopword]
        keyword_list = self.noun_tokens
        self.freqtxt = pd.Series(dict(FreqDist(keyword_list))).sort_values(ascending=False)
        c2 = collections.Counter(keyword_list)
        a = c2.most_common(50)
        file = open('../static/data/news_threeDays_mining.csv', 'w', encoding='utf-8', newline='')
        print(file.name)
        csvfile = csv.writer(file)
        for row in a:
            csvfile.writerow(row)
        file.close()
        return file

if __name__ == '__main__':
    service = Service()
    result = service.tokenize()
    print(result)

# ==============================================================
# ====================                     =====================
# ====================    Preprocessing    =====================
# ====================                     =====================
# ==============================================================


class StockDf(object):
    '''
	* @ ihidNum attribute 를 리턴한다.
	* @ return String
    * @ Read_csv_create_wordcloud:   
    '''
    def read(self):
        with open('data/30_news_threeDays_mining.csv', 'r', encoding='utf-8') as f:
            data = csv.reader(f)
            t = list()
            for x, y in data:
                aa = dict(text=x, value=y)
                print(aa)
                print(type(aa))
                t.append(aa)
            p = json.dumps(t)
        return p

    def propensity_classify(self, period):
    
        corp_total = pd.read_excel('data/재무제표.xlsx')
        corp_symbol = list(np.array(corp_total['종목코드'].tolist()))

        ticker_list = []

        for symbol in corp_symbol:
            ticker_list.append(str(symbol).zfill(6))

        date = ''
        if (period == '단기'):
            date = str(datetime.now() + timedelta(days=-30))
        if (period == '중기'):
            date = str(datetime.now() + timedelta(days=-180))
        if (period == '중장기'):
            date = '2019'
        if (period == '장기'):
            date = '2017'

        df_list = [fdr.DataReader(ticker, date)['Close'] for ticker in ticker_list]
        df = pd.concat(df_list, axis=1)
        df.columns = list(np.array(corp_total['종목명'].tolist()))
        df = df.dropna()

        slope = {}

        for company in df.columns:
            close = list(np.array(df[company].tolist()))
            x1 = min(close)
            y1 = close.index(x1)
            x2 = max(close)
            y2 = close.index(x2)
            a = abs((x2 - x1) / (y2 - y1))
            slope[company] = a

        companies = []

        for y, v in sorted(slope.items(), key=lambda slope: slope[1]):
            companies.append(y)

        propensity = {'안정형': companies[0:10], '안정추구형': companies[10:20],
                      '위험중립형': companies[20:30], '적극투자형': companies[30:40],
                      '공격투자형': companies[40:]}
        return propensity

    def new_magic_formula(self, period, propensity):
        corp_total = pd.read_excel('data/재무제표.xlsx')
        companies = []
        if (propensity == '안정형'):
            companies = self.propensity_classify(period)['안정형']
        if (propensity == '안정추구형'):
            companies = self.propensity_classify(period)['안정추구형']
        if (propensity == '위험중립형'):
            companies = self.propensity_classify(period)['위험중립형']
        if (propensity == '적극투자형'):
            companies = self.propensity_classify(period)['적극투자형']
        if (propensity == '공격투자형'):
            companies = self.propensity_classify(period)['공격투자형']

        corp_total = corp_total[corp_total['종목명'].isin(companies)]
        corp_total = corp_total.dropna()

        ticker_list = []
        name_list = []
        recommendation_dic = {}

        if (period == '단기'):
            corp_total['2020/06 GP/A'] = corp_total['2020/06 매출총이익'] / corp_total['2020/06 자본총계']
            corp_total['2020/06 BPS'] = corp_total['2020/06 자본총계'] * 100000000 / corp_total['상장주식수']
            corp_total['2020/06 PBR'] = corp_total['현재가'] / corp_total['2020/06 BPS']
            corp_total['2020/06 GP/A Rank'] = corp_total['2020/06 GP/A'].rank()
            corp_total['2020/06 PBR Rank'] = corp_total['2020/06 PBR'].rank(ascending=0)
            corp_total['Total Rank'] = corp_total['2020/06 GP/A Rank'] + corp_total['2020/06 PBR Rank']
            corp_recommendation = corp_total.sort_values(by=['Total Rank'])[:5]
            symbol_list = list(np.array(corp_recommendation['종목코드'].tolist()))
            for symbol in symbol_list:
                ticker_list.append(str(symbol).zfill(6))
            name_list = list(np.array(corp_recommendation['종목명'].tolist()))
            recommendation_dic = {'종목코드': ticker_list, '종목명': name_list}

        if (period == '중기'):
            corp_total['2020/06 GP/A'] = corp_total['2020/06 매출총이익'] / corp_total['2020/06 자본총계']
            corp_total['2020/06 BPS'] = corp_total['2020/06 자본총계'] * 100000000 / corp_total['상장주식수']
            corp_total['2020/06 PBR'] = corp_total['현재가'] / corp_total['2020/06 BPS']
            corp_total['2020/03 GP/A'] = corp_total['2020/03 매출총이익'] / corp_total['2020/03 자본총계']
            corp_total['2020/03 BPS'] = corp_total['2020/03 자본총계'] * 100000000 / corp_total['상장주식수']
            corp_total['2020/03 PBR'] = corp_total['현재가'] / corp_total['2020/03 BPS']
            corp_total['2020/06 GP/A Rank'] = corp_total['2020/06 GP/A'].rank()
            corp_total['2020/06 PBR Rank'] = corp_total['2020/06 PBR'].rank(ascending=0)
            corp_total['2020/03 GP/A Rank'] = corp_total['2020/03 GP/A'].rank()
            corp_total['2020/03 PBR Rank'] = corp_total['2020/03 PBR'].rank(ascending=0)
            corp_total['Total Rank'] = corp_total['2020/06 GP/A Rank'] + corp_total['2020/06 PBR Rank'] \
                                       + corp_total['2020/03 GP/A Rank'] + corp_total['2020/03 PBR Rank']
            corp_recommendation = corp_total.sort_values(by=['Total Rank'])[:5]
            symbol_list = list(np.array(corp_recommendation['종목코드'].tolist()))
            for symbol in symbol_list:
                ticker_list.append(str(symbol).zfill(6))
            name_list = list(np.array(corp_recommendation['종목명'].tolist()))
            recommendation_dic = {'종목코드': ticker_list, '종목명': name_list}

        if (period == '중장기'):
            corp_total['2020/06 GP/A'] = corp_total['2020/06 매출총이익'] / corp_total['2020/06 자본총계']
            corp_total['2020/06 BPS'] = corp_total['2020/06 자본총계'] * 100000000 / corp_total['상장주식수']
            corp_total['2020/06 PBR'] = corp_total['현재가'] / corp_total['2020/06 BPS']
            corp_total['2020/03 GP/A'] = corp_total['2020/03 매출총이익'] / corp_total['2020/03 자본총계']
            corp_total['2020/03 BPS'] = corp_total['2020/03 자본총계'] * 100000000 / corp_total['상장주식수']
            corp_total['2020/03 PBR'] = corp_total['현재가'] / corp_total['2020/03 BPS']
            corp_total['2019 GP/A'] = corp_total['2019 매출총이익'] / corp_total['2019 자본총계']
            corp_total['2019 BPS'] = corp_total['2019 자본총계'] * 100000000 / corp_total['상장주식수']
            corp_total['2019 PBR'] = corp_total['현재가'] / corp_total['2019 BPS']
            corp_total['2020/06 GP/A Rank'] = corp_total['2020/06 GP/A'].rank()
            corp_total['2020/06 PBR Rank'] = corp_total['2020/06 PBR'].rank(ascending=0)
            corp_total['2020/03 GP/A Rank'] = corp_total['2020/03 GP/A'].rank()
            corp_total['2020/03 PBR Rank'] = corp_total['2020/03 PBR'].rank(ascending=0)
            corp_total['2019 GP/A Rank'] = corp_total['2019 GP/A'].rank()
            corp_total['2019 PBR Rank'] = corp_total['2019 PBR'].rank(ascending=0)
            corp_total['Total Rank'] = corp_total['2020/06 GP/A Rank'] + corp_total['2020/06 PBR Rank'] \
                                       + corp_total['2020/03 GP/A Rank'] + corp_total['2020/03 PBR Rank'] \
                                       + corp_total['2019 GP/A Rank'] + corp_total['2019 PBR Rank']
            corp_recommendation = corp_total.sort_values(by=['Total Rank'])[:5]
            symbol_list = list(np.array(corp_recommendation['종목코드'].tolist()))
            for symbol in symbol_list:
                ticker_list.append(str(symbol).zfill(6))
            name_list = list(np.array(corp_recommendation['종목명'].tolist()))
            recommendation_dic = {'종목코드': ticker_list, '종목명': name_list}

        if (period == '장기'):
            corp_total['2020/06 GP/A'] = corp_total['2020/06 매출총이익'] / corp_total['2020/06 자본총계']
            corp_total['2020/06 BPS'] = corp_total['2020/06 자본총계'] * 100000000 / corp_total['상장주식수']
            corp_total['2020/06 PBR'] = corp_total['현재가'] / corp_total['2020/06 BPS']
            corp_total['2020/03 GP/A'] = corp_total['2020/03 매출총이익'] / corp_total['2020/03 자본총계']
            corp_total['2020/03 BPS'] = corp_total['2020/03 자본총계'] * 100000000 / corp_total['상장주식수']
            corp_total['2020/03 PBR'] = corp_total['현재가'] / corp_total['2020/03 BPS']
            corp_total['2019 GP/A'] = corp_total['2019 매출총이익'] / corp_total['2019 자본총계']
            corp_total['2019 BPS'] = corp_total['2019 자본총계'] * 100000000 / corp_total['상장주식수']
            corp_total['2019 PBR'] = corp_total['현재가'] / corp_total['2019 BPS']
            corp_total['2018 GP/A'] = corp_total['2018 매출총이익'] / corp_total['2018 자본총계']
            corp_total['2018 BPS'] = corp_total['2018 자본총계'] * 100000000 / corp_total['상장주식수']
            corp_total['2018 PBR'] = corp_total['현재가'] / corp_total['2018 BPS']
            corp_total['2017 GP/A'] = corp_total['2017 매출총이익'] / corp_total['2017 자본총계']
            corp_total['2017 BPS'] = corp_total['2017 자본총계'] * 100000000 / corp_total['상장주식수']
            corp_total['2017 PBR'] = corp_total['현재가'] / corp_total['2017 BPS']
            corp_total['2020/06 GP/A Rank'] = corp_total['2020/06 GP/A'].rank()
            corp_total['2020/06 PBR Rank'] = corp_total['2020/06 PBR'].rank(ascending=0)
            corp_total['2020/03 GP/A Rank'] = corp_total['2020/03 GP/A'].rank()
            corp_total['2020/03 PBR Rank'] = corp_total['2020/03 PBR'].rank(ascending=0)
            corp_total['2019 GP/A Rank'] = corp_total['2019 GP/A'].rank()
            corp_total['2019 PBR Rank'] = corp_total['2019 PBR'].rank(ascending=0)
            corp_total['2018 GP/A Rank'] = corp_total['2018 GP/A'].rank()
            corp_total['2018 PBR Rank'] = corp_total['2018 PBR'].rank(ascending=0)
            corp_total['2017 GP/A Rank'] = corp_total['2017 GP/A'].rank()
            corp_total['2017 PBR Rank'] = corp_total['2017 PBR'].rank(ascending=0)
            corp_total['Total Rank'] = corp_total['2020/06 GP/A Rank'] + corp_total['2020/06 PBR Rank'] \
                                       + corp_total['2020/03 GP/A Rank'] + corp_total['2020/03 PBR Rank'] \
                                       + corp_total['2019 GP/A Rank'] + corp_total['2019 PBR Rank'] \
                                       + corp_total['2018 GP/A Rank'] + corp_total['2018 PBR Rank'] \
                                       + corp_total['2017 GP/A Rank'] + corp_total['2017 PBR Rank']
            corp_recommendation = corp_total.sort_values(by=['Total Rank'])[:5]
            symbol_list = list(np.array(corp_recommendation['종목코드'].tolist()))
            for symbol in symbol_list:
                ticker_list.append(str(symbol).zfill(6))
            name_list = list(np.array(corp_recommendation['종목명'].tolist()))

            recommendation_dic = {'종목코드': ticker_list, '종목명': name_list}

        return recommendation_dic

    def recommendation_listing(self, period, propensity):
        recommendation_dic = self.new_magic_formula(period=period, propensity=propensity)
        ticker_list = recommendation_dic['종목코드']
        now_price = []
        change = []
        change_ratio = []
        symbol = []

        for ticker in ticker_list:
            date = datetime.today().strftime('%Y-%m-%d')
            df = fdr.DataReader(ticker, date)
            prev = df.iloc[0, 0]
            now = df.iloc[0, 3]
            now_price.append(str(now))
            change_won = ''
            if prev - now > 0:
                change_won = '▲ {}'.format(str(prev - now))
            elif prev-now == 0:
                change_won = 0
            else:
                change_won = '▼ {}'.format(str(abs(prev - now)))
            change.append(change_won)
            change_ratio.append('{}%'.format(round((prev - now) / prev * 100, 2)))

        result = {'종목명': recommendation_dic['종목명'], '현재가': now_price, '전일대비': change, '전일비': change_ratio, '종목코드': ticker_list}

        return result




# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================

class StockDto(db.Model):
    ...

class StockVo(object):
    ...

class StockDao(db.Model):
    ...

# ==============================================================
# =======================                  =====================
# =======================    Service       =====================
# =======================                  =====================
# ==============================================================
class StockTF(object):
    ...


class StockService(object):
    ...



# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================

class Stock(Resource):
    '''
    @app.route('/recommendation/<period>/<propensity>', methods=['GET'])
    def recommend_stock(period, propensity):
        period = period
        propensity = propensity
        print(period, propensity)
        recommend_stock_model = Recommendation_Stock_Model()
        return json.dumps(recommend_stock_model.recommendation_listing(period=period, propensity=propensity))


    @app.route('/cloud', methods=['GET'])
    def create_wordcloud_using_csv():
        C = Read_csv_create_wordcloud()
        result = C.read()
        print(result)
        return result


    @app.route('/stocks/candle/<symbol>', methods=['GET'])
    def getCandle(symbol):
        symbol = symbol
        a = candleController()
        app_result = a.candle_crawling(symbol=symbol)
        return json.dumps(app_result)


    CORS(app)
    if __name__ == '__main__':
        app.run()
    '''