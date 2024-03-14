import flask
from flask import Blueprint, request, redirect, url_for
from flask_login import login_user, login_required, logout_user, current_user

from extensions.db import get_session

from models import Order, OrderItem, CartItem, Drink, User

carts_blueprint = Blueprint('carts', __name__)


@carts_blueprint.route('/cart', methods=['GET'])
@login_required
def cart():
    with get_session() as session:
        cart_items = session.query(CartItem).filter_by(
            user_id=current_user.id).all()
        drinks = session.query(Drink).all()
        return flask.render_template('cart.html', cart_items=cart_items, drinks=drinks)


@carts_blueprint.route('/cart/add', methods=['POST'])
@login_required
def add_to_cart():
    drink_id = request.form['drink_id']
    quantity = request.form['quantity']

    with get_session() as session:
        cart_item = CartItem(
            user_id=current_user.id,
            drink_id=drink_id,
            quantity=quantity
        )
        session.add(cart_item)
        session.commit()

    return redirect(url_for('carts.cart'))


@carts_blueprint.route('/cart/remove', methods=['POST'])
@login_required
def remove_from_cart():
    cart_item_id = request.form['cart_item_id']

    with get_session() as session:
        cart_item = session.query(CartItem).filter_by(
            id=cart_item_id).first()
        session.delete(cart_item)
        session.commit()

    return redirect(url_for('carts.cart'))


@carts_blueprint.route('/cart/checkout', methods=['POST'])
@login_required
def checkout():
    with get_session() as session:
        cart_items = session.query(CartItem).filter_by(
            user_id=current_user.id).all()

        order = Order(user_id=current_user.id)
        session.add(order)
        session.commit()

        for cart_item in cart_items:
            order_item = OrderItem(
                order_id=order.id,
                drink_id=cart_item.drink_id,
                quantity=cart_item.quantity
            )
            session.add(order_item)
            session.delete(cart_item)
            session.commit()

    return redirect(url_for('carts.cart'))
