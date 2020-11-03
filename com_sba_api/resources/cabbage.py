from typing import List
from flask import request
from flask_restful import Resource, reqparse
from flask import jsonify
from com_sba_api.ext.db import db, openSession, engine
from pathlib import Path
from sqlalchemy import func
from dataclasses import dataclass
import json
import pandas as pd
import json
import os
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


@dataclass
class FileReader:
    # def __init__(self, context, fname, train, test, id, label):
    #     self._context = context  # _ 1ea default access, _ 2ea private access

    # 3.7부터 간소화되서 dataclass 데코 후, key: value 형식으로 써도 됨 (롬복 형식)
    context : str = ''
    fname: str = ''
    train: object = None
    test: object = None
    id : str = ''
    lable : str = ''
    

    def new_file(self) -> str:
        return os.path.join(self.context,self.fname)

    def csv_to_dframe(self) -> object:
        return pd.read_csv(self.new_file(), encoding='UTF-8', thousands=',')

    def xls_to_dframe(self, header, usecols) -> object:
        print(f'PANDAS VERSION: {pd.__version__}')
        return pd.read_excel(self.new_file(), header = header, usecols = usecols)

    def json_load(self):
        return json.load(open(self.new_file(), encoding='UTF-8'))


# ==============================================================
# ====================                     =====================
# ====================    Preprocessing    =====================
# ====================                     =====================
# ==============================================================


class CabbageDf(object):
    def __init__(self):
        self.fileReader = FileReader()  
        self.data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
        self.df = None

    def new_train(self, payload) -> object:
        this = self.fileReader
        this.data = self.data
        this.fname = payload
        print(f'{self.data}')
        print(f'{this.fname}')
        return pd.read_csv(Path(self.data, this.fname)) 

    def new(self):
        this = self.fileReader
        price_data = 'price_data.csv'
        this.train = self.new_train(price_data)
        print(this.train.columns)
        '''
        Index(['year', 'avgTemp', 'minTemp', 'maxTemp', 'rainFall', 'avgPrice'], dtype='object')
        '''
        self.df = pd.DataFrame(

            {
             'year' : this.train.year,
             'avg_temp' : this.train.avgTemp,
             'min_temp' : this.train.minTemp,
             'max_temp' : this.train.maxTemp,
             'rain_fall' : this.train.rainFall,
             'avg_price' : this.train.avgPrice
             }
        )
        return self.df
    
    
    
    

'''
CabbageDF.new()
          year  avgTemp  minTemp  maxTemp  rainFall  avgPrice
0     20100101     -4.9    -11.0      0.9       0.0      2123
1     20100102     -3.1     -5.5      5.5       0.8      2123
2     20100103     -2.9     -6.9      1.4       0.0      2123
3     20100104     -1.8     -5.1      2.2       5.9      2020
4     20100105     -5.2     -8.7     -1.8       0.7      2060
...        ...      ...      ...      ...       ...       ...
2917  20171227     -3.9     -8.0      0.7       0.0      2865
2918  20171228     -1.5     -6.9      3.7       0.0      2884
2919  20171229      2.9     -2.1      8.0       0.0      2901
2920  20171230      2.9     -1.6      7.1       0.6      2901
2921  20171231      2.1     -2.0      5.8       0.4      2901

[2922 rows x 6 columns]


from flask import Flask
app = Flask(__name__)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

config = {
    'user' : 'root',
    'password' : 'root',
    'host': '127.0.0.1',
    'port' : '3306',
    'database' : 'com_sba_api'
}
charset = {'utf8':'utf8'}
url = f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?charset=utf8"
Base = declarative_base()
engine = create_engine(url)
app.config['SQLALCHEMY_DATABASE_URI'] = url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
db.init_app(app)

       year  avg_temp  min_temp  max_temp  rain_fall  avg_price
0  20100101      -4.9     -11.0       0.9        0.0       2123
1  20100102      -3.1      -5.5       5.5        0.8       2123
2  20100103      -2.9      -6.9       1.4        0.0       2123
3  20100104      -1.8      -5.1       2.2        5.9       2020
4  20100105      -5.2      -8.7      -1.8        0.7       2060
'''


# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================


class CabbageDto(db.Model):
    __tablename__='cabbages'
    __table_args__={'mysql_collate':'utf8_general_ci'}

    year: str = db.Column(db.String(10), primary_key = True, index = True)
    avg_temp: float = db.Column(db.Float)
    min_temp: float = db.Column(db.Float)
    max_temp: float = db.Column(db.Float)
    rain_fall: float = db.Column(db.Float)
    avg_price: int = db.Column(db.Integer)

    def __init__(self, year, avg_temp, min_temp, max_temp, rain_fall, avg_price):
        self.year = year
        self.avg_temp = avg_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.rain_fall = rain_fall
        self.avg_price = avg_price

    def __repr__(self):
        return f'Cabbage(year= {self.year}, avg_temp={self.avg_temp}, min_temp={self.min_temp}\
            , max_temp={self.max_temp}, rain_fall={self.rain_fall}, avg_price={self.avg_price})'

 
'''
with app.app_context():
    db.create_all()
    db.session.commit()
'''


class CabbageVo:
    year: str = ''
    avg_temp: float = 0.0
    min_temp: float = 0.0
    max_temp: float = 0.0
    rain_fall: float = 0.0
    avg_price: int = 0




Session = openSession()
session = Session()
cabbage_df = CabbageDf()

class CabbageDao(CabbageDto):

    @staticmethod
    def bulk():
        Session = openSession()
        session = Session()
        cabbage_df = CabbageDf()
        df = cabbage_df.hook()
        print(df.head())
        session.bulk_insert_mappings(CabbageDto, df.to_dict(orient='records'))
        session.commit()
        session.close()

    @staticmethod
    def count():
        return session.query(func.count(CabbageDto.year)).one()

    @staticmethod
    def save(cabbage):
        new_cabbage = CabbageDto(year= cabbage['year'],
                                avg_temp= cabbage['avg_temp'],
                                min_temp= cabbage['min_temp'],
                                max_temp= cabbage['max_temp'],
                                rain_fall= cabbage['rain_fall'],
                                avg_price= cabbage['avg_price'])
        session.add(new_cabbage)
        session.commit() 


class CabbageTf(object):

    def __init__(self):
        self.path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models','cabbage')

    def new(self):
        
        tf.global_variables_initializer()
        df = pd.read_sql_table('cabbages', engine.connect())
        xy = np.array(df, dtype=np.float32)
        x_data = xy[:,1:-1] #feature
        y_data = xy[:,[-1]] # 가격
        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        hypothesis = tf.matmul(X, W) + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # 러닝
        for step in range(100000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={X: x_data, Y: y_data})
            if step % 500 == 0:
                print('# %d 손실비용 : %d'%(step, cost_))
                print("- 배추가격 : %d" % (hypo_[0]))
                
        # 저장
        saver = tf.train.Saver()
        
        saver.save(sess, self.path+'/cabbage.ckpt')
        print('저장완료')


# ==============================================================
# =======================                  =====================
# =======================    Service       =====================
# =======================                  =====================
# ==============================================================

class CabbageService(object):
    def __init__(self):
        self.path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models','cabbage2')
       
    avg_temp: float = 0.0
    min_temp: float = 0.0
    max_temp: float = 0.0
    rain_fall: float = 0.0

    def assign(self, param):
        self.avg_temp = param.avg_temp
        self.min_temp = param.min_temp
        self.max_temp = param.max_temp
        self.rain_fall = param.rain_fall

    def predict(self):
        X = tf.placeholder(tf.float32, shape=[None, 4])
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver.restore(sess, self.path+'/cabbage.ckpt')
            data = [[self.avg_temp, self.min_temp, self.max_temp, self.rain_fall],]
            arr = np.array(data, dtype = np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:4]})
            print(dict[0])
        return int(dict[0])



if __name__ == "__main__":
    c = CabbageTf()
    c.new()

    '''
    service = CabbageService()
    cabbage = CabbageVo()
    cabbage.avg_temp = 10
    cabbage.max_temp = -5
    cabbage.min_temp = 30
    cabbage.rain_fall = 20
    service.assign(cabbage)
    price = service.predict()
    print(f'Predicted Cabbage Price is {price} won')
    '''


# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================




parser = reqparse.RequestParser()
parser.add_argument('avg_temp', type=str, required=True,
                                        help='This field should be a userId')
parser.add_argument('min_temp', type=str, required=True,
                                        help='This field should be a password')
parser.add_argument('max_temp', type=str, required=True,
                                        help='This field should be a password')                                        
parser.add_argument('rain_fall', type=str, required=True,
                                        help='This field should be a password')
class Cabbage(Resource):
        
    @staticmethod
    def post():
        service = CabbageService()
        args = parser.parse_args()
        cabbage = CabbageVo()
        cabbage.avg_temp = args.avg_temp
        cabbage.max_temp = args.max_temp
        cabbage.min_temp = args.min_temp
        cabbage.rain_fall = args.rain_fall
        service.assign(cabbage)
        price = service.predict()
        print(f'Predicted Cabbage Price is {price} won')
        return {'price': str(price)}, 200 
        
