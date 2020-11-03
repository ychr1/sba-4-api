from flask import Flask
from flask_restful import Api
from com_sba_api.ext.db import url, db
from com_sba_api.ext.routes import initialize_routes
from com_sba_api.resources.user import UserDao
from com_sba_api.resources.cabbage import CabbageDao
from flask_cors import CORS



app = Flask(__name__)
CORS(app, resources={r'/api/*': {"origins": "*"}})

app.config['SQLALCHEMY_DATABASE_URI'] = url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
api = Api(app)
with app.app_context():
    db.create_all()
    user_count = UserDao.count()
    print(f'***** Users Total Count is {user_count} *****')
    if user_count[0] == 0:
        UserDao.bulk()

    cabb_count = CabbageDao.count()
    print(f'***** Cabbages Total Count is {cabb_count} *****')
    if cabb_count[0] == 0:
        CabbageDao.bulk()

initialize_routes(api)


