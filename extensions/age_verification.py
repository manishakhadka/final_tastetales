from flask_login import login_required, current_user
from flask import redirect, url_for
from functools import wraps


def age_verified(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_adult:
            return redirect(url_for('age_verification.not_verified'))
        return f(*args, **kwargs)
    return decorated_function


def age_not_verified(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_user.is_adult:
            return redirect(url_for('age_verification.verified'))
        return f(*args, **kwargs)
    return decorated_function
