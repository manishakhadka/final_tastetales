from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, UserMixin, current_user

from extensions.db import get_session, get_db
from models import User, Drink

from routes.user import users_blueprint
from routes.drinks import drinks_blueprint
from routes.cart import carts_blueprint
from routes.order import orders_blueprint
from routes.age_verification import age_verification_blueprint
from routes.admin import admins_blueprint

app = Flask(__name__)
app.secret_key = 'K>KPP@uSfA*P#:E2[K{/J~oJ[R9R,qyv0^i]~Jo`@SvH]1//c:Y&P:SLzRi(`H)'

login_manager = LoginManager()
login_manager.init_app(app)

# Register blueprints
app.register_blueprint(users_blueprint)
app.register_blueprint(drinks_blueprint)
app.register_blueprint(carts_blueprint)
app.register_blueprint(orders_blueprint)
app.register_blueprint(age_verification_blueprint)
app.register_blueprint(admins_blueprint)


@login_manager.user_loader
def user_loader(user_id):
    session = get_session()
    user = session.query(User).filter_by(id=user_id).first()
    session.close()
    return user


@login_manager.unauthorized_handler
def unauthorized():
    return redirect(url_for('users.login'))


@app.route('/')
def index():
    is_adult = False
    if current_user.is_authenticated:
        is_adult = current_user.is_adult

    with get_session() as session:
        if is_adult:
            drinks = session.query(Drink).all()
        else:
            drinks = session.query(Drink).filter_by(drink_type='soft').all()
        return render_template('index.html', drinks=drinks)


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/healthcheck')
def healthcheck():
    return 'OK'


if __name__ == '__main__':
    app.run(debug=True)
