from wtforms import Form, StringField, PasswordField, validators


class RegisterForm(Form):
    first_name = StringField(
        'First Name', [validators.DataRequired(), validators.Length(min=4, max=25)])
    middle_name = StringField(
        'Middle Name',)
    last_name = StringField(
        'Last Name', [validators.DataRequired(), validators.Length(min=4, max=25)])
    email = StringField(
        'Email Address', [validators.DataRequired(), validators.Length(min=6, max=35)])
    password = PasswordField('New Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')


class LoginForm(Form):
    email = StringField('Your Email Address', [
                        validators.DataRequired(),
                        validators.Length(min=6, max=35)])
    password = PasswordField('Password', [validators.DataRequired()])
