from functools import wraps
from flask import redirect, url_for
from flask_login import current_user
import flask


def login_not_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_user.is_authenticated:
            next = flask.request.args.get('next')
            return redirect(next or url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.role == 'admin':
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function
