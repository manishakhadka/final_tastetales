import flask

from flask import Blueprint, request, redirect, url_for
from models import User
from extensions.db import get_session
from extensions.age_verification import age_verified, age_not_verified
from utils.camera import process_image_and_predict_age

from flask_login import login_required, current_user

age_verification_blueprint = Blueprint('age_verification', __name__)


@age_verification_blueprint.route('/verify-age', methods=['GET', 'POST'])
@login_required
@age_not_verified
def verify_age():
    if request.method == 'POST':
        # if 'file' not in request.files:
        #     flask.flash('No file part')
        #     return redirect(request.url)
        # file = request.files['file']
        # if file.filename == '':
        #     flask.flash('No image found. Please capture again.')
        #     return redirect(request.url)
        file = request.form['image_data']
        if file:
            age = process_image_and_predict_age(file)
            with get_session() as session:
                user = session.query(User).filter_by(
                    id=current_user.id).first()
                user.is_adult = age >= 18
                user.is_age_verified = True
                user.predicted_age = age
                session.commit()
            return redirect(url_for('age_verification.verified'))
    return flask.render_template('age_verification.html')


@age_verification_blueprint.route('/age-not-verified', methods=['GET'])
@login_required
@age_not_verified
def not_verified():
    return flask.render_template('age_not_verified.html')


@age_verification_blueprint.route('/age-verified', methods=['GET'])
@login_required
@age_verified
def verified():
    return flask.render_template('age_verified.html')
