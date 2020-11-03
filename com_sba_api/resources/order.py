from flask_restful import Resource
from flask import Response, jsonify
from com_sba_api.ext.db import db
from flask_restful import Resource, reqparse
from com_sba_api.ext.db import db, openSession
import json
from com_sba_api.resources.user import UserDto
from com_sba_api.resources.item import ItemDto
from com_sba_api.resources.order import OrderDto
from com_sba_api.resources.price import PriceDto
from sqlalchemy import func
from datetime import date

Session = openSession()
session = Session()

# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================

class OrderDto(db.Model):
    
       __tablename__ = 'orders'

       order_id = db.Column(db.Integer, primary_key=True)
       order_quantity = db.Column(db.Integer)
       order_price = db.Column(db.Integer)
       order_delivery = db.Column(db.Date)
       order_placed = db.Column(db.Date)

       user_id = db.Column(db.String(10), db.ForeignKey(UserDto.user_id))
       user = db.relationship('UserDto', back_populates='orders')
       item_id = db.Column(db.Integer, db.ForeignKey(ItemDto.item_id))
       item = db.relationship('ItemDto', back_populates='orders')
       

       def __init__(self, user_id, item_id, 
                    order_price, order_quantity,
                    order_delivery, order_placed):
           self.user_id = user_id
           self.item_id = item_id
           self.order_quantity = order_quantity
           self.order_price = order_price
           self.order_delivery = order_delivery
           self.order_placed = order_placed

       def __repr__(self):
           return '<Order {0}>'.format(self.orderDelivery)


class OrderDao(OrderDto):

 
    '''
    SELECT * FROM items
        JOIN prices ON prices.itemId=items.id
            WHERE prices.userId = 1 AND prices.available = True
        LEFT JOIN (SELECT * FROM orders WHERE order_delivery = '2017-07-05') as orders
             ON orders.itemId=items.id

    query = db.session.query(Item, Price, Order).\
        from_statement(db.text("""
                                SELECT * FROM items
                                    JOIN prices ON prices.itemId=items.id
                                    LEFT JOIN (
                                        SELECT * FROM orders 
                                        WHERE orderDelivery = :order_delivery) as orders
                                            ON orders.itemId=items.id
                                WHERE prices.userId = :userId AND prices.available
                                """)).\
                                    params(userId=1, orderDelivery='2017-07-05')
                                
    '''
    '''@classmethod
    def find_specific_day_orders(cls, user):
        orders = session.query(OrderDto).\
            filter(OrderDto.order_delivery == date(2017, 7, 5)).subquery()
                
        orders_alias = db.aliased(OrderDto, orders)
        session.query(ItemDto, PriceDto, orders_alias).\
            join(PriceDto).\
                outerjoin(orders_alias, ItemDto.orders).\
                    filter(PriceDto.user_id == 1,PriceDto.available).all()'''



# ==============================================================
# =======================                  =====================
# =======================    Resource    =======================
# =======================                  =====================
# ==============================================================

parser = reqparse.RequestParser()  
parser.add_argument('userId', type=str, required=True,
                                        help='This field should be a userId')
parser.add_argument('password', type=str, required=True,
                                        help='This field should be a password')

class Order(Resource):
    ...

class Orders(Resource):
   
    @staticmethod
    def get():
        try:
            orders = OrderDao.find_specific_day_orders()
            if orders:
                return orders.json()
        except Exception as e:
            return {'Orders': 'User not found'}, 404

