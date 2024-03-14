import flask
from flask import Blueprint, request
from models import Order
from extensions.db import get_session
from extensions.auth import login_not_required
from forms.user import LoginForm, RegisterForm

from flask_login import login_user, logout_user, login_required, current_user

admins_blueprint = Blueprint('admins', __name__)


@admins_blueprint.route('/admin', methods=['GET', 'POST'])
@login_required
def admin_products():
    with get_session() as session:
        orders = session.query(Order).all()
        return flask.render_template('admin.html', orders=orders)
    return flask.render_template('admin.html')
