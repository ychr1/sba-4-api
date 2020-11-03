from flask_restful import Resource
from flask import Response, jsonify
from com_sba_api.ext.db import db
from com_sba_api.resources.item import ItemDto
from com_sba_api.resources.user import UserDto
# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================
'''
class PriceDto(db.Model):
    
       __tablename__ = 'prices'

       price_id = db.Column(db.Integer, primary_key=True)
       price = db.Column(db.Float)

       user_id = db.Column(db.String(10), db.ForeignKey(UserDto.user_id))
       user = db.relationship('UserDto', back_populates='prices')
       item_id = db.Column(db.Integer, db.ForeignKey(ItemDto.item_id))
       item = db.relationship('ItemDto', back_populates='prices')

       def __init__(self, userId, itemId, priceMeasurement, price):
           self.userId = userId
           self.itemId = itemId
           self.priceMeasurement = priceMeasurement
           self.price = price

       def __repr__(self):
           return '<Price {0}>'.format(self.price)
'''
# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================
