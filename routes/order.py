import flask
from flask import Blueprint, request, redirect, url_for
from flask_login import login_user, login_required, logout_user, current_user

from extensions.db import get_session

from models import Order, OrderItem, CartItem, Drink, User

orders_blueprint = Blueprint('orders', __name__)


@orders_blueprint.route('/orders', methods=['GET'])
@login_required
def orders():
    with get_session() as session:
        orders = session.query(Order).filter_by(
            user_id=current_user.id).all()
        return flask.render_template('orders.html', orders=orders)
