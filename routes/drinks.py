import flask
from flask import Blueprint, request, redirect, url_for
from flask_login import login_user, login_required, logout_user, current_user

from extensions.db import get_session
from extensions.auth import admin_required

from models import Order, OrderItem, CartItem, Drink, User

drinks_blueprint = Blueprint('drinks', __name__)


@drinks_blueprint.route('/drinks', methods=['GET'])
def drinks():
    with get_session() as session:
        drinks = session.query(Drink).all()
        return flask.render_template('drinks.html', drinks=drinks)


@drinks_blueprint.route('/drinks/recommended', methods=['GET'])
def recommended():
    with get_session() as session:
        drinks = session.query(Drink).all()
        return flask.render_template('drinks.html', drinks=drinks)


@drinks_blueprint.route('/drinks/softdrinks', methods=['GET'])
def softdrinks():
    with get_session() as session:
        drinks = session.query(Drink).filter_by(category='softdrink').all()
        return flask.render_template('drinks.html', drinks=drinks)


@drinks_blueprint.route('/drinks/harddrinks', methods=['GET'])
@login_required
def harddrinks():
    if not current_user.is_adult:
        return redirect(url_for('age_verification.verify_age'))
    with get_session() as session:
        drinks = session.query(Drink).filter_by(category='harddrink').all()
        return flask.render_template('drinks.html', drinks=drinks)


@drinks_blueprint.route('/drinks/add', methods=['GET', 'POST'])
@admin_required
def add_drink():
    if request.method == 'POST':
        name = request.form['name']
        category = request.form['category']
        price = request.form['price']
        with get_session() as session:
            drink = Drink(name=name, category=category, price=price)
            session.add(drink)
            session.commit()
        return redirect(url_for('drinks.drinks'))
    return flask.render_template('add_drink.html')


@drinks_blueprint.route('/drinks/edit', methods=['GET', 'POST'])
@admin_required
def edit_drink():
    drink_id = request.args['drink_id']
    with get_session() as session:
        drink = session.query(Drink).filter_by(id=drink_id).first()
        if request.method == 'POST':
            drink.name = request.form['name']
            drink.category = request.form['category']
            drink.price = request.form['price']
            session.commit()
            return redirect(url_for('drinks.drinks'))
    return flask.render_template('edit_drink.html', drink=drink)


@drinks_blueprint.route('/drinks/remove', methods=['POST'])
@admin_required
def remove_drink():
    drink_id = request.form['drink_id']
    with get_session() as session:
        drink = session.query(Drink).filter_by(id=drink_id).first()
        session.delete(drink)
        session.commit()
    return redirect(url_for('drinks.drinks'))


@drinks_blueprint.route('/drinks/order', methods=['POST'])
@login_required
def order_drink():
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

    return redirect(url_for('drinks.drinks'))


@drinks_blueprint.route('/drinks/order/checkout', methods=['POST'])
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

    return redirect(url_for('drinks.drinks'))
