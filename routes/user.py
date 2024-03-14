import flask
from flask import Blueprint, request
from models import User
from extensions.db import get_session
from extensions.auth import login_not_required
from forms.user import LoginForm, RegisterForm

from flask_login import login_user, logout_user, login_required, current_user

users_blueprint = Blueprint('users', __name__)


@users_blueprint.route('/login', methods=['GET', 'POST'])
@login_not_required
def login():
    form = LoginForm(request.form)
    if request.method == 'POST' and form.validate():

        with get_session() as session:
            user: User = session.query(User).filter_by(
                email=form.email.data).first()
            if user is None:
                flask.flash('User not found. Please register.')
                return flask.redirect(flask.url_for('users.login'))

            if not user.check_password(form.password.data):
                flask.flash('Incorrect password')
                return flask.redirect(flask.url_for('users.login'))

            login_user(user)

        flask.flash('Logged in successfully.')

        next = flask.request.args.get('next')

        default_route = 'admins.admin_products' if user.role == 'admin' else 'index'

        return flask.redirect(next or flask.url_for(default_route))
    else:
        for fieldName, errorMessages in form.errors.items():
            for err in errorMessages:
                flask.flash(f"Error in {fieldName}: {err}")
    return flask.render_template('login.html', form=form)


@users_blueprint.route('/register', methods=['GET', 'POST'])
@login_not_required
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        with get_session() as session:
            user = User(email=form.email.data, password=form.password.data,
                        first_name=form.first_name.data, last_name=form.last_name.data,
                        is_adult=False, is_age_verified=False, role='user')
            session.add(user)
            session.commit()
        flask.flash('Thanks for registering')
        return flask.redirect(flask.url_for('users.login'))
    else:
        for fieldName, errorMessages in form.errors.items():
            for err in errorMessages:
                flask.flash(f"Error in {fieldName}: {err}")
    return flask.render_template('register.html', form=form)


@users_blueprint.route('/logout')
@login_required
def logout():
    logout_user()
    return flask.redirect(flask.url_for('index'))


@users_blueprint.route('/profile')
@login_required
def profile():
    return flask.render_template('profile.html', name=current_user.name)
