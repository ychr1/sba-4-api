import logging
from flask import Blueprint
from flask_restful import Api
from com_sba_api.resources.home import Home
from com_sba_api.resources.item import Item, Items
from com_sba_api.resources.user import User, Users, Auth, Access
from com_sba_api.resources.article import Article, Articles
from com_sba_api.resources.cabbage import Cabbage

home = Blueprint('home', __name__, url_prefix='/api')
user = Blueprint('user', __name__, url_prefix='/api/user')
users = Blueprint('users', __name__, url_prefix='/api/users')
auth = Blueprint('auth', __name__, url_prefix='/api/auth')
access = Blueprint('access', __name__, url_prefix='/api/access')
article = Blueprint('article', __name__, url_prefix='/api/article')
articles = Blueprint('articles', __name__, url_prefix='/api/articles')
cabbage = Blueprint('cabbage', __name__, url_prefix='/api/cabbage')

api = Api(home)
api = Api(user)
api = Api(users)
api = Api(auth)
api = Api(access)
api = Api(article)
api = Api(articles)

def initialize_routes(api):
    
    api.add_resource(Home, '/api')
    api.add_resource(Item, '/api/item/<string:id>')
    api.add_resource(Items,'/api/items')
    api.add_resource(User, '/api/user/<string:id>')
    api.add_resource(Users, '/api/users')
    api.add_resource(Auth, '/api/auth')
    api.add_resource(Access, '/api/access')
    api.add_resource(Article, '/api/article')
    api.add_resource(Articles, '/api/articles/')
    api.add_resource(Cabbage, '/api/cabbage')

@user.errorhandler(500)
def user_api_error(e):
    logging.exception('An error occurred during user request. %s' % str(e))
    return 'An internal error occurred.', 500

@home.errorhandler(500)
def home_api_error(e):
    logging.exception('An error occurred during home request. %s' % str(e))
    return 'An internal error occurred.', 500

@article.errorhandler(500)
def article_api_error(e):
    logging.exception('An error occurred during article request. %s' % str(e))
    return 'An internal error occurred.', 500

@cabbage.errorhandler(500)
def cabbage_api_error(e):
    logging.exception('An error occurred during cabbage request. %s' % str(e))
    return 'An internal error occurred.', 500