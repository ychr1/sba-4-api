
from main import app
app.run(host='127.0.0.1', port='8080', debug=True)

'''
from flask import Flask
from flask_restful import Resource, Api
 
app = Flask(__name__)
api = Api(app)
 
class Rest(Resource):
    def get(self):
        return {'rest': 'Good !'}
    def post(self):
        return {'rest': 'post success !'}
 
api.add_resource(Rest, '/api')
 
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
'''