from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, TIMESTAMP, text, JSON, ForeignKey, Boolean
from sqlalchemy.dialects.mysql import INTEGER, DOUBLE, DECIMAL
from sqlalchemy.orm import relationship, DeclarativeBase, mapped_column, Mapped

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

Base = declarative_base()
metadata = Base.metadata


class User(Base, UserMixin):

    __tablename__ = 'users'

    id = Column(INTEGER(11), primary_key=True)
    first_name = Column(String(255))
    last_name = Column(String(255))
    email = Column(String(255))
    password = Column(String(255))
    is_adult = Column(Boolean, default=False)
    is_age_verified = Column(Boolean, default=False)
    predicted_age = Column(INTEGER(11))
    role = Column(String(255), default='user')  # user | admin
    created_on = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP"))

    def __init__(self, first_name, last_name, email, password, is_adult=False, is_age_verified=False, role='user'):
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.password = generate_password_hash(password)
        self.is_adult = is_adult
        self.is_age_verified = is_age_verified
        self.role = role

    def is_active(self):
        # Here we should write whatever the code is
        # that checks the database if our user is active
        return self.active

    def is_anonymous(self):
        return False

    def is_authenticated(self):
        return True

    def get_id(self):
        return self.id

    def check_password(self, password):
        return check_password_hash(self.password, password)


class Drink(Base):

    __tablename__ = 'drinks'

    id = Column(INTEGER(11), primary_key=True)
    name = Column(String(255))
    image = Column(String(255))
    price = Column(INTEGER(11))
    discount = Column(DOUBLE, default=0)
    description = Column(String(255))

    available_quantity = Column(INTEGER(11), default=0)

    drink_type = Column(String(255))  # soft | hard

    created_on = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP"))
    updated_on = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class CartItem(Base):

    __tablename__ = 'cart_items'

    id = Column(INTEGER(11), primary_key=True)
    user_id = Column(INTEGER(11), ForeignKey('users.id'))
    drink_id = Column(INTEGER(11), ForeignKey('drinks.id'))
    quantity = Column(INTEGER(11))
    created_on = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP"))
    updated_on = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

    user = relationship('User', backref='cart_items')
    drink = relationship('Drink', backref='cart_items')


class Order(Base):

    __tablename__ = 'orders'

    id = Column(INTEGER(11), primary_key=True)
    user_id = Column(INTEGER(11), ForeignKey('users.id'))
    total_price = Column(INTEGER(11))

    message = Column(String(255))
    address = Column(String(255))
    contact = Column(String(255))

    # cash | khalti | esewa | net banking
    payment_mode = Column(String(255), default='cash')

    status = Column(String(255))  # pending | completed
    created_on = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP"))
    updated_on = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

    user = relationship('User', backref='orders')


class OrderItem(Base):

    __tablename__ = 'order_items'

    id = Column(INTEGER(11), primary_key=True)
    order_id = Column(INTEGER(11), ForeignKey('orders.id'))
    drink_id = Column(INTEGER(11), ForeignKey('drinks.id'))
    quantity = Column(INTEGER(11), default=1)
    created_on = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP"))
    updated_on = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

    order = relationship('Order', backref='order_items')
    drink = relationship('Drink', backref='order_items')


class Token(Base):

    __tablename__ = 'tokens'

    id = Column(INTEGER(11), primary_key=True)
    user_id = Column(INTEGER(11), ForeignKey('users.id'))
    token = Column(String(255))
    created_on = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP"))
    updated_on = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

    user = relationship('User', backref='tokens')
