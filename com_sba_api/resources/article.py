from flask_restful import Resource, reqparse
from com_sba_api.ext.db import db, openSession
from com_sba_api.resources.user import UserDto
from com_sba_api.resources.item import ItemDto

class ArticleDto(db.Model):
    __tablename__ = "articles"
    __table_args__={'mysql_collate':'utf8_general_ci'}

    art_id: int = db.Column(db.Integer, primary_key=True, index=True)
    title: str = db.Column(db.String(100))
    content: str = db.Column(db.String(500))

    user_id = db.Column(db.String(10), db.ForeignKey(UserDto.user_id))
    user = db.relationship('UserDto', back_populates='articles')
    item_id = db.Column(db.Integer, db.ForeignKey(ItemDto.item_id))
    item = db.relationship('ItemDto', back_populates='articles')

    def __init__(self, title, content, user_id, item_id):
        self.title = title
        self.content = content
        self.user_id = user_id
        self.item_id = item_id

    def __repr__(self):
        return f'art_id={self.art_id}, user_id={self.user_id}, item_id={self.item_id},\
            title={self.title}, content={self.content}'

    def json(self):
        return {
            'art_id': self.art_id,
            'user_id': self.user_id,
            'item_id' : self.item_id,
            'title' : self.title,
            'content' : self.content
        }
class ArticleVo():
    art_id: int = 0
    user_id: str = ''
    item_id: int = 0
    title: str = ''
    content: str = ''

class ArticleDao(ArticleDto):
    
    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_name(cls, name):
        return cls.query.filer_by(name == name).all()

    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter(ArticleDto.art_id == id).one()

    @staticmethod
    def save(article):
        Session = openSession()
        session = Session()
        session.add(article)
        session.commit()

    @staticmethod
    def update(article, article_id):
        Session = openSession()
        session = Session()
        session.query(ArticleDto).filter(ArticleDto.art_id == article.article_id)\
            .update({ArticleDto.title: article.title,
                        ArticleDto.content: article.content})
        session.commit()

    @classmethod
    def delete(cls,art_id):
        Session = openSession()
        session = Session()
        cls.query(ArticleDto.art_id == art_id).delete()
        session.commit()

            



class Article(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()
        
    def post(self):
        parser = self.parser
        parser.add_argument('user_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('item_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('title', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('content', type=str, required=False, help='This field cannot be left blank')
        args = parser.parse_args()
        article = ArticleDto(args['title'], args['content'],\
                            args['user_id'], args['item_id'])
        try: 
            ArticleDao.save(article)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200    
        except:
            return {'message': 'An error occured inserting the article'}, 500
    @staticmethod
    def get(id):
        article = ArticleDao.find_by_id(id)
        if article:
            return article.json()
        return {'message': 'Article not found'}, 404
    @staticmethod
    def put(self, article, article_id):
        parser = self.parser
        parser.add_argument('art_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('user_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('item_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('title', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('content', type=str, required=False, help='This field cannot be left blank')
        args = parser.parse_args()
        article = ArticleVo()
        article.title = args['title']
        article.content = args['content']
        article.art_id = args['art_id']
        try: 
            ArticleDao.update(article, article_id)
            return {'message': 'Article was Updated successfully'}, 200
        except:
            return {'message': 'An error occured updating the article'}, 500


class Articles(Resource):
    def get(self):
        return {'articles': list(map(lambda article: article.json(), ArticleDao.find_all()))}
        # return {'articles':[article.json() for article in ArticleDao.find_all()]}



    