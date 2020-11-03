from typing import List
from flask import request
from flask_restful import Resource, reqparse
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
from com_sba_api.util.file import FileReader
from flask import jsonify
from com_sba_api.ext.db import db, openSession
import pandas as pd
import json
import os
import pandas as pd
import numpy as np

"""
context: /Users/bitcamp/SbaProjects
fname: 
PassengerId
Survived: The answer that a machine learning model should match 
Pclass: Boarding Pass 1 = 1st-class seat, 2 = 2nd, 3 = 3rd,
Name,
Sex,
Age,
SibSp accompanying brothers, sisters, spouses
Parch accompanying parents, children,
Ticket : Ticket Number
Fare : Boarding Charges
Cabin : Room number
Embarked : a Port Name on Board C = Cherbourg, Q = Queenstown, S = Southhampton
"""   
# ==============================================================
# ====================                     =====================
# ====================    Preprocessing    =====================
# ====================                     =====================
# ==============================================================

class UserDf(object):
    def __init__(self):
        self.fileReader = FileReader()  
        self.data = os.path.join(os.path.abspath(os.path.dirname(__file__))+'\\data')
        self.odf = None

    def new(self):
        train = 'train.csv'
        test = 'test.csv'
        this = self.fileReader
        this.train = self.new_model(train) # payload
        this.test = self.new_model(test) # payload
        
        '''
        Original Model Generation
        '''
        self.odf = pd.DataFrame(

            {
             'user_id' : this.train.PassengerId,
             'password' : '1',
             'name' : this.train.Name
             }
        )
        
        this.id = this.test['PassengerId'] # This becomes a question. 
        # print(f'Preprocessing Train Variable : {this.train.columns}')
        # print(f'Preprocessing Test Variable : {this.test.columns}')
        this = self.drop_feature(this, 'Cabin')
        this = self.drop_feature(this, 'Ticket')
        # print(f'Post-Drop Variable : {this.train.columns}')
        this = self.embarked_norminal(this)
        # print(f'Preprocessing Embarked Variable: {this.train.head()}')
        this = self.title_norminal(this)
        # print(f'Preprocessing Title Variable: {this.train.head()}')
        '''
        The name is unnecessary because we extracted the Title from the name variable.
        '''
        this = self.drop_feature(this, 'Name')
        this = self.drop_feature(this, 'PassengerId')
        this = self.age_ordinal(this)
        # print(f'Preprocessing Age Variable: {this.train.head()}')
        this = self.drop_feature(this, 'SibSp')
        this = self.sex_norminal(this)
        # print(f'Preprocessing Sex Variable: {this.train.head()}')
        this = self.fareBand_nominal(this)
        # print(f'Preprocessing Fare Variable: {this.train.head()}')
        this = self.drop_feature(this, 'Fare')
        # print(f'Preprocessing Train Result: {this.train.head()}')
        # print(f'Preprocessing Test Result: {this.test.head()}')
        # print(f'Train NA Check: {this.train.isnull().sum()}')
        # print(f'Test NA Check: {this.test.isnull().sum()}')
        this.label = self.create_label(this) # payload
        this.train = self.create_train(this) # payload
        # print(f'Train Variable : {this.train.columns}')
        # print(f'Test Variable : {this.train.columns}')
        clf = RandomForestClassifier()
        clf.fit(this.train, this.label)
        prediction = clf.predict(this.test)
        
        # print(this)
        df = pd.DataFrame(

            {
             'pclass': this.train.Pclass,
             'gender': this.train.Sex, 
             'age_group': this.train.AgeGroup,
             'embarked' : this.train.Embarked,
             'rank' : this.train.Title
             }
        )
     
        # print(self.odf)
        # print(df)
        sumdf = pd.concat([self.odf, df], axis=1)

        return sumdf
        
    
    def new_model(self, payload) -> object:
        this = self.fileReader
        this.data = self.data
        this.fname = payload
        print(f'{self.data}')
        print(f'{this.fname}')
        return pd.read_csv(Path(self.data, this.fname)) 

    @staticmethod
    def create_train(this) -> object:
        return this.train.drop('Survived', axis=1) # Train is a dataset in which the answer is removed. 

    @staticmethod
    def create_label(this) -> object:
        return this.train['Survived'] # Label is the answer.

    @staticmethod
    def drop_feature(this, feature) -> object:
        this.train = this.train.drop([feature], axis = 1)
        this.test = this.test.drop([feature], axis = 1) 
        return this


    @staticmethod
    def pclass_ordinal(this) -> object:
        return this

    @staticmethod
    def sex_norminal(this) -> object:
        combine = [this.train, this.test] # Train and test are bound.
        sex_mapping = {'male':0, 'female':1}
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)
        this.train = this.train # overriding
        this.test = this.test
        return this

    @staticmethod
    def age_ordinal(this) -> object:
        train = this.train
        test = this.test 
        train['Age'] = train['Age'].fillna(-0.5)
        test['Age'] = test['Age'].fillna(-0.5)
        '''
        It's ambiguous to put an average, and it's too baseless to put a majority.
        the age is significant in determining survival rates and requires a detailed approach.
        If you don't know your age, 
        you have to deal with it without knowing it to reduce the distortion of the price
        -0.5 is the middle value.
        '''
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf] 
        '''
        This part represents a range.
        -1 and more than 0....60 and more...
        [] This must be a variable name here.If you think so, you've got it right.
        '''
         
        labels = ['Unknown', 'Baby', 'Child', 'Teenager','Student','Young Adult', 'Adult', 'Senior']
        # [] This must be a variable name here.
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        test['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        age_title_mapping = {
            0: 'Unknown',
            1: 'Baby',
            2: 'Child',
            3: 'Teenager',
            4: 'Student',
            5: 'Young Adult',
            6: 'Adult',
            7: 'Senior'
        } # If you treat it from [] to {} like this, you will treat Labs as a value.
        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]
        for x in range(len(test['AgeGroup'])):
            if test['AgeGroup'][x] == 'Unknown':
                test['AgeGroup'][x] = age_title_mapping[test['Title'][x]]
        
        age_mapping = {
            'Unknown': 0,
            'Baby': 1,
            'Child': 2,
            'Teenager': 3,
            'Student': 4,
            'Young Adult': 5,
            'Adult': 6,
            'Senior': 7
        }
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
        this.train = train
        this.test = test
        return this

    @staticmethod
    def sibsp_numeric(this) -> object:
        return this

    @staticmethod
    def parch_numeric(this) -> object:
        return this

    @staticmethod
    def fare_ordinal(this) -> object:
        this.train['FareBand'] = pd.qcut(this['Fare'], 4, labels={1,2,3,4})
        this.test['FareBand'] = pd.qcut(this['Fare'], 4, labels={1,2,3,4})
        return this


    @staticmethod
    def fareBand_nominal(this) -> object:  # Rates vary, so prepare for clustering
        this.train = this.train.fillna({'FareBand' : 1})  # FareBand is a non-existent variable added
        this.test = this.test.fillna({'FareBand' : 1})
        return this

    @staticmethod
    def embarked_norminal(this) -> object:
        this.train = this.train.fillna({'Embarked': 'S'}) # S is the most common, filling in empty spaces.
        this.test = this.test.fillna({'Embarked': 'S'}) 
        '''
        Many machine learning libraries expect class labels to be encoded as * integer*
        mapping: blue = 0, green = 1, red = 2
        '''
        this.train['Embarked'] = this.train['Embarked'].map({'S': 1, 'C' : 2, 'Q' : 3}) 
        this.test['Embarked'] = this.test['Embarked'].map({'S': 1, 'C' : 2, 'Q' : 3})
        return this

    @staticmethod
    def title_norminal(this) -> object:
        combine = [this.train, this.test]
        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
        for dataset in combine:
            dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev',\
                'Jonkheer','Dona', 'Mme'], 'Rare')
            dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'], 'Royal')
            dataset['Title'] = dataset['Title'].replace('Ms','Miss')
            dataset['Title'] = dataset['Title'].replace('Mlle','Mr')
        title_mapping = {'Mr':1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0) # Unknown
        this.train = this.train
        this.test = this.test
        return this

    # Dtree, rforest, nb, nnn, svm among Learning Algorithms use this as a representative

    @staticmethod
    def create_k_fold():
        return KFold(n_splits=10, shuffle=True, random_state=0)
    

    def accuracy_by_dtree(self, this):
        dtree = DecisionTreeClassifier()
        score = cross_val_score(dtree, this.train, this.label, cv=UserDf.create_k_fold(),\
             n_jobs=1, scoring='accuracy')
        return round(np.mean(score) * 100, 2)

    def accuracy_by_rforest(self, this):
        rforest = RandomForestClassifier()
        score = cross_val_score(rforest, this.train, this.label, cv=UserDf.create_k_fold(), \
            n_jobs=1, scoring='accuracy')
        return round(np.mean(score) * 100, 2)
    
    def accuracy_by_nb(self, this):
        nb = GaussianNB()
        score = cross_val_score(nb, this.train, this.label, cv=UserDf.create_k_fold(),\
             n_jobs=1, scoring='accuracy')
        return round(np.mean(score) * 100, 2)
    
    def accuracy_by_knn(self, this):
        knn = KNeighborsClassifier()
        score = cross_val_score(knn, this.train, this.label, cv=UserDf.create_k_fold(),\
             n_jobs=1, scoring='accuracy')
        return round(np.mean(score) * 100, 2)

    def accuracy_by_svm(self, this):
        svm = SVC()
        score = cross_val_score(svm, this.train, this.label, cv=UserPreprocess.create_k_fold(),\
             n_jobs=1, scoring='accuracy')
        return round(np.mean(score) * 100, 2)

    def learning(self, train, test):
        service = self.service
        this = self.modeling(train, test)
        print(f'Dtree verification result: {service.accuracy_by_dtree(this)}')
        print(f'RForest verification result: {service.accuracy_by_rforest(this)}')
        print(f'Naive Bayes tree verification result: {service.accuracy_by_nb(this)}')
        print(f'KNN verification result: {service.accuracy_by_knn(this)}')
        print(f'SVM verification result: {service.accuracy_by_svm(this)}')

    def submit(self, train, test): 
        this = self.modeling(train, test)
        clf = RandomForestClassifier()
        clf.fit(this.train, this.label)
        prediction = clf.predict(this.test)
        
        print(this)
        # Pclass  Sex   Age  Parch  Embarked  Title AgeGroup
        df = pd.DataFrame(

            {
             'pclass': this.train.Pclass,
             'gender': this.train.Sex, 
             'age_group': this.train.AgeGroup,
             'embarked' : this.train.Embarked,
             'rank' : this.train.Title
             }
        )
      
        # print(self.odf)
        # print(df)
        sumdf = pd.concat([self.odf, df], axis=1)
        print(sumdf)
        return sumdf
            







'''
user_id password                                               name  pclass  gender age_group  embarked  rank
0         1        1                            Braund, Mr. Owen Harris       3       0         4         1     1
1         2        1  Cumings, Mrs. John Bradley (Florence Briggs Th...       1       1         6         2     3
2         3        1                             Heikkinen, Miss. Laina       3       1         5         1     2
3         4        1       Futrelle, Mrs. Jacques Heath (Lily May Peel)       1       1         5         1     3
4         5        1                           Allen, Mr. William Henry       3       0         5         1     1
..      ...      ...                                                ...     ...     ...       ...       ...   ...
886     887        1                              Montvila, Rev. Juozas       2       0         5         1     6
887     888        1                       Graham, Miss. Margaret Edith       1       1         4         1     2
888     889        1           Johnston, Miss. Catherine Helen "Carrie"       3       1         2         1     2
889     890        1                              Behr, Mr. Karl Howell       1       0         5         2     1
890     891        1                                Dooley, Mr. Patrick       3       0         5         3     1
[891 rows x 8 columns]
'''

# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================

class UserDto(db.Model):

    __tablename__ = 'users'
    __table_args__={'mysql_collate':'utf8_general_ci'}

    user_id: str = db.Column(db.String(10), primary_key = True, index = True)
    password: str = db.Column(db.String(1))
    name: str = db.Column(db.String(100))
    pclass: int = db.Column(db.Integer)
    gender: int = db.Column(db.Integer)
    age_group: int = db.Column(db.Integer)
    embarked: int = db.Column(db.Integer)
    rank: int = db.Column(db.Integer)

    # orders = db.relationship('OrderDto', back_populates='user', lazy='dynamic')
    # prices = db.relationship('PriceDto', back_populates='user', lazy='dynamic')
    articles = db.relationship('ArticleDto', back_populates='user', lazy='dynamic')

    def __init__(self, user_id, password, name, pclass, gender, age_group, embarked, rank):
        self.user_id = user_id
        self.password = password
        self.name = name
        self.pclass = pclass
        self.gender = gender
        self.age_group = age_group
        self.embarked = embarked
        self.rank = rank

    def __repr__(self):
        return f'User(user_id={self.user_id},\
            password={self.password},name={self.name}, pclass={self.pclass}, gender={self.gender}, \
                age_group={self.age_group}, embarked={self.embarked}, rank={self.rank})'

    
    def __str__(self):
        return f'User(user_id={self.user_id},\
            password={self.password},name={self.name}, pclass={self.pclass}, gender={self.gender}, \
                age_group={self.age_group}, embarked={self.embarked}, rank={self.rank})'


    
    def json(self):
        return {
            'userId' : self.user_id,
            'password' : self.password,
            'name' : self.name,
            'pclass' : self.pclass,
            'gender' : self.gender,
            'ageGroup' : self.age_group,
            'embarked' : self.embarked,
            'rank' : self.rank
        }
   

    
class UserVo:
    user_id: str = ''
    password: str = ''
    name: str = ''
    pclass: int = 0
    gender: int = 0
    age_group: int = 0
    embarked: int = 0
    rank: int =  0

Session = openSession()
session = Session()
user_df = UserDf()

class UserDao(UserDto):

    @staticmethod   
    def bulk():
        df = user_df.new()
        print(df.head())
        session.bulk_insert_mappings(UserDto, df.to_dict(orient="records"))
        session.commit()
        session.close()

    @staticmethod
    def count():
        return session.query(func.count(UserDto.user_id)).one()

    @staticmethod
    def save(user):
        db.session.add(user)
        db.session.commit()

    @staticmethod
    def update(user):
        db.session.add(user)
        db.session.commit()

    @classmethod
    def delete(cls,id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    
    '''
    SELECT *
    FROM users
    WHERE user_name LIKE 'a'
    '''
    # like() method itself produces the LIKE criteria 
    # for WHERE clause in the SELECT expression.
    
    @classmethod
    def find_one(cls, user_id):
        
        # return session.query(UserDto).filter(UserDto.user_id.like(user_id)).one()
        query = cls.query\
            .filter(cls.user_id.like(user_id))
        df = pd.read_sql(query.statement, query.session.bind)
        print('>>>>>>>>>>>>>>>>>')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    
    '''
    SELECT *
    FROM users
    WHERE user_name LIKE 'name'
    '''
    # the meaning of the symbol %
    # A% ==> Apple
    # %A ==> NA
    # %A% ==> Apple, NA, BAG 
    @classmethod
    def find_by_name(cls, name):
        return session.query(UserDto).filter(UserDto.user_id.like(f'%{name}%')).all()

    '''
    SELECT *
    FROM users
    WHERE user_id IN (start, end)
    '''
    # List of users from start to end ?
    @classmethod
    def find_users_in_category(cls, start, end):
        return session.query(UserDto)\
                      .filter(UserDto.user_id.in_([start,end])).all()

    '''
    SELECT *
    FROM users
    WHERE gender LIKE 'gender' AND name LIKE 'name%'
    '''
    # Please enter this at the top. 
    # from sqlalchemy import and_
    @classmethod
    def find_users_by_gender_and_name(cls, gender, name):
        return session.query(UserDto)\
                      .filter(and_(UserDto.gender.like(gender), UserDto.name.like(f'{name}%'))).all()

    '''
    SELECT *
    FROM users
    WHERE pclass LIKE '1' OR age_group LIKE '3'
    '''
    # Please enter this at the top. 
    # from sqlalchemy import or_
    @classmethod
    def find_users_by_gender_and_name(cls, gender, age_group):
        return session.query(UserDto)\
                      .filter(or_(UserDto.pclass.like(gender), UserDto.age_group.like(f'{age_group}%'))).all()
    
    
    @classmethod
    def login(cls, user):
        return session.query(cls)\
            .filter(cls.user_id == user.user_id, 
            cls.password == user.password)\
            .one()
            


if __name__ == "__main__":
    UserDao.bulk()



# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================


parser = reqparse.RequestParser()  # only allow price changes, no name changes allowed


class User(Resource):
    @staticmethod
    def post():
        parser.add_argument('id')
        parser.add_argument('password')
        args = parser.parse_args()
        print(f'User {args["id"]} added ')
        params = json.loads(request.get_data(), encoding='utf-8')
        if len(params) == 0:

            return 'No parameter'

        params_str = ''
        for key in params.keys():
            params_str += 'key: {}, value: {}<br>'.format(key, params[key])
        return {'code':0, 'message': 'SUCCESS'}, 200

    @staticmethod
    def get(id: str):
        try:
            user = UserDao.find_one(id)
            if user:
                print(f' !!!!!!!!!!!!!!{user} ')
                return json.dumps(user), 200
        except Exception as e:
            return {'message': 'User not found'}, 404


    @staticmethod
    def update():
        parser.add_argument('userId')
        parser.add_argument('password')
        args = parser.parse_args()
        print(f'User {args["id"]} updated ')
        return {'code':0, 'message': 'SUCCESS'}, 200

    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'USer {args["id"]} deleted')
        return {'code' : 0, 'message' : 'SUCCESS'}, 200    

class Users(Resource):
    @staticmethod
    def post():
        ud = UserDao()
        ud.bulk('users')
    @staticmethod
    def get():
        data = UserDao.find_all()
        return data, 200

class Auth(Resource):
    @staticmethod
    def post():
        body = request.get_json()
        user = UserDto(**body)
        UserDao.save(user)
        id = user.user_id
        
        return {'id': str(id)}, 200 

'''
json = json.loads() => dict
dict = json.dumps() => json
'''
class Access(Resource):
    @staticmethod
    def post():
        parser.add_argument('id')
        parser.add_argument('password')
        args = parser.parse_args()
        user = UserVo()
        user.user_id = args.id
        user.password = args.password
        print(user.user_id)
        print(user.password)
        data = UserDao.login(user)
        print(f'Login Result : {data}')
        return data.json(), 200






