# Code to predict age using cv2 and cnn model

# age_detection_code.py

from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from flask import jsonify
from age_detection_code import age_model
from age_detection_code import face_cascade
import mysql.connector
from imp import load_module
import cv2
import random
import numpy as np
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

# custom cnn code
from train import (
    CHECKPOINTS_DIR,
    NUM_AGE_CATEGORIES,
    AGE_CATEGORY_MAP,
    initialize_model,
    parse_filepath,
    load_checkpoint_and_predict
)

# Load the pre-trained age detection model
FCAES_DIR = "faces"

age_model = initialize_model()

# Open the webcam
cap = cv2.VideoCapture(0)

# Define the face detection cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the face image for the age model


def preprocess_face_for_age_model(face_image):
    # Resize the face image to match the expected input shape of the model
    face_image = cv2.resize(face_image, (200, 200))
    face_image = face_image / 255.0  # Normalize pixel values between 0 and 1
    # Add an additional dimension to represent the single channel (grayscale)
    face_image = np.expand_dims(face_image, axis=-1)
    return face_image

# Function to postprocess the age prediction


def postprocess_age_prediction(age_prediction):
    # Find the index with the highest probability
    predicted_age_index = np.argmax(age_prediction)

    # Convert the index to an age range using your custom function
    predicted_age = class_labels_reassign(predicted_age_index)

    return predicted_age

# Function to reassign age labels to ranges


def class_labels_reassign(age_label):
    if age_label == 0:
        return "1-2"
    elif age_label == 1:
        return "3-9"
    elif age_label == 2:
        return "10-17"
    elif age_label == 3:
        return "18-27"
    elif age_label == 4:
        return "28-45"
    elif age_label == 5:
        return "46-65"
    else:
        return "66-100"


def save_and_get_path(face_image, face_number):
    face_path = FCAES_DIR + "/face_" + str(face_number) + ".jpg"
    cv2.imwrite(face_path, face_image)
    return face_path


def get_permission_and_age():
    print("Attempting to access the webcam...")
    # Check if the webcam is accessible
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        return -1  # or handle the error accordingly

    # Add code to request permission from the user
    # Example: permission_result = input("Do you give permission to access the age detection feature? (yes/no): ")
    permission_result = "yes"  # Replace with actual code

    if permission_result.lower() == 'yes':
        # Capture frame from webcam and process age detection
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Couldn't read frame from webcam.")
            return -1  # or handle the error accordingly

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)
        print("faces", faces)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_input = preprocess_face_for_age_model(face_roi)
            face_number = random.randint(1, 1000)
            face_path = save_and_get_path(face_roi, face_number)
            # age_prediction = age_model.predict(face_input.reshape(1, *face_input.shape))
            print("face_path", face_path)
            y_pred = load_checkpoint_and_predict(
                age_model, CHECKPOINTS_DIR, face_path)
            y_pred_age, y_pred_category = y_pred['age_output'], y_pred['age_category_output']
            predicted_age_int = int(y_pred_age[0][0])
            cap.release()
            return predicted_age_int

            # predicted_age_range = postprocess_age_prediction(age_prediction)
            # return predicted_age_range
    else:
        # Return a value indicating that permission was denied
        print("Permission denied!")
        return -1


# if __name__ == "__main__":
#     age = get_permission_and_age()
#     if age != -1:
#         print(f"The predicted age range is: {age}")
#     else:
#         print("Error: Age detection failed.")
#     cap.release()

# Release the webcam and close the OpenCV windows
# cap.release()
# cv2.destroyAllWindows()


# Web Application code


# from age_detection_code import get_permission_and_age

# recommendation ko lagi import
# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')


# Function to process age detection and return age prediction
# app.py


@app.route('/detect_age', methods=['POST'])
def detect_age():
    permission_result = request.form.get('age_verification.verify_age')
    if permission_result == 'yes':
        # Use the modified get_permission_and_age function
        age = get_permission_and_age()
        print("Predicted Age:", age)

        if age == -1 or age is None:
            # Redirect to a page indicating no permission
            return "Permission denied! You cannot access this feature."

        # Extract the last number from the age range string
        # age_str = str(age)
        # last_age_number = int(age_str.split('-')[-1])

        if age > 18:
            return redirect(url_for('harddrinks'))
        else:
            return redirect(url_for('underage'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/softdrinks')
def softdrinks():
    return render_template('softdrinks.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


# Replace these credentials with your MySQL database credentials
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'tastetales',
    # Use 'mysql_native_password' for MySQL 8
    'auth_plugin': 'mysql_native_password'
}


@app.route('/addToCart', methods=['POST'])
def addToCart():
    if request.method == 'POST':
        # Access form data using request.form
        image_url = request.form['image_url']
        product_name = request.form['product_name']
        # Ensure it's an integer
        total_quantity = int(request.form['total_quantity'])
        total_price = float(request.form['total_price'])  # Ensure it's a float

        # Connect to the MySQL database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Insert data into the "cart" table
        sql = "INSERT INTO cart (image_url, product_name, quantity, price) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (image_url, product_name,
                       total_quantity, total_price))

        # Commit the changes to the database
        conn.commit()

        # Close the database connection
        conn.close()

        # Redirect to the cart page
        return redirect(url_for('cartpage'))


@app.route('/cartpage')
def cartpage():
    # Render the cart page template
    return render_template('cartpage.html')


@app.route('/order')
def order():

    return render_template('order.html')


@app.route('/deleteCartItem', methods=['POST'])
def deleteCartItem():
    # Get the item ID from the POST request
    item_id = request.form['itemId']

    # Connect to the database
    servername = "localhost"
    username = "root"
    password = ""
    dbname = "tastetales"

    conn = pymysql.connect(host=servername, user=username,
                           password=password, database=dbname)

    try:
        with conn.cursor() as cursor:
            # Delete the item from the database
            sql = f"DELETE FROM cart WHERE id = {item_id}"
            cursor.execute(sql)
            conn.commit()
            return jsonify({'status': 'success', 'message': 'Item deleted successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error deleting item: {str(e)}'})
    finally:
        conn.close()


@app.route('/permission')
def permission():
    return render_template('permission.html')

# @app.route('/permission_response/<response>')
# def permission_response(response):
#     # Check if the user granted permission
#     if response.lower() == 'yes':
#         # Redirect to the age detection route
#         return redirect(url_for('age_detection'))
#     else:
#         # Handle denial of permission (if needed)
#         return "Permission denied"


# @app.route('/age_detection')
# def age_detection():
#     # Redirect to a page with JavaScript to handle age detection in the browser
#     return render_template('age_detection.html')


# import numpy as np
# import base64

# @app.route('/predict_age', methods=['POST'])
# def predict_age():
#     data = request.get_json()
#     image_data = data.get('image')

#     # Decode base64-encoded image string to bytes
#     image_bytes = base64.b64decode(image_data.split(',')[1])

#     # Process image_bytes to detect age
#     # For demonstration purposes, let's assume a dummy age prediction
#     age_range = '18-25'
#     return jsonify({'age_range': age_range})


@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')


@app.route('/harddrinks')
def harddrinks():
    return render_template('harddrinks.html')


@app.route('/underage')
def underage():
    return render_template('underage.html')


# Load your DataFrame
df1 = pd.read_excel(r'Nepal_cheers_Liquor_Online_ALcololic_Beveragee.xlsx')

# TF-IDF vectorizer for name recommendations
tfidf_vectorizer_name = TfidfVectorizer(
    stop_words='english', max_df=0.85, min_df=0.05, max_features=500)
name_data = df1['Name']
tfidf_matrix_name = tfidf_vectorizer_name.fit_transform(name_data)
cosine_similarities_name = linear_kernel(tfidf_matrix_name, tfidf_matrix_name)

# TF-IDF vectorizer for brand recommendations
brand_vectorizer = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
brand_matrix = brand_vectorizer.fit_transform(df1['Brand'])
brand_cosine_similarities = linear_kernel(brand_matrix, brand_matrix)

# Numeric features matrix for price recommendations
numeric_features = df1[['Price', 'Alcohol']]
scaler = MinMaxScaler()
numeric_features_normalized = scaler.fit_transform(numeric_features)
combined_matrix_price = np.concatenate(
    [cosine_similarities_name, numeric_features_normalized], axis=1)


@app.route('/get_name_recommendations', methods=['POST'])
def get_name_recommendations():
    user_input_name = request.json['name']
    print(f"User Input Name: {user_input_name}")

    # Ensure the 'Name' column is converted to lowercase before comparison
    input_index = df1.index[df1['Name'].str.lower(
    ) == user_input_name.lower()].tolist()

    if not input_index:
        print(f"Name not found: {user_input_name}")
        return jsonify([])

    input_index = input_index[0]
    print(f"Input Index: {input_index}")

    name_similarity_scores = list(
        enumerate(cosine_similarities_name[input_index]))
    sorted_names = sorted(name_similarity_scores,
                          key=lambda x: x[1], reverse=True)
    top5_recommendations = [df1['Name'][i[0]] for i in sorted_names[1:6]]

    print(f"Top 5 Recommendations: {top5_recommendations}")

    return jsonify(top5_recommendations)


@app.route('/get_brand_recommendations', methods=['POST'])
def get_brand_recommendations():
    user_input_brand = request.json['brand']
    print(f"User Input Brand: {user_input_brand}")

    # Ensure the 'Brand' column is converted to lowercase before comparison
    user_brand_index = df1.index[df1['Brand'].str.lower(
    ) == user_input_brand.lower()].tolist()

    if not user_brand_index:
        print(f"Brand not found: {user_input_brand}")
        return jsonify([])

    user_brand_index = user_brand_index[0]
    print(f"User Brand Index: {user_brand_index}")

    brand_similarities = combined_matrix_price[user_brand_index]
    print(f"Brand Similarities: {brand_similarities}")

    similar_brands_indices = brand_similarities.argsort()[-6:-1][::-1]
    print(f"Similar Brands Indices: {similar_brands_indices}")

    similar_brands = df1.iloc[similar_brands_indices]['Brand']
    recommended_products = df1.iloc[similar_brands_indices]['Name']

    print(f"Similar Brands: {similar_brands}")
    print(f"Recommended Products: {recommended_products}")

    return jsonify(list(recommended_products))


@app.route('/get_price_recommendations', methods=['POST'])
def get_price_recommendations():
    user_input_price = request.json['price']

    print(f"Received Price: {user_input_price}")

    try:
        user_input_price = int(user_input_price)
    except ValueError:
        print("Invalid price format.")
        return jsonify([])

    # Find products with prices similar to or higher than the user input
    similar_products = df1[df1['Price'] >=
                           user_input_price].sort_values(by='Price').head(5)

    if similar_products.empty:
        print("No similar products found.")
        return jsonify([])

    recommended_products = list(similar_products['Name'])

    print(f"Recommended Products: {recommended_products}")

    return jsonify(recommended_products)


# php files ko lagi
# admin ko lagi
@app.route('/adminlogin')
def adminlogin():
    return render_template('adminlogin.html')


# databaselai
# MySQL configurations
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'tastetales'
}

# Connect to MySQL database
db_connection = mysql.connector.connect(**mysql_config)


@app.route('/insert_data', methods=['POST'])
def insert_data():
    if request.method == 'POST':
        image_url = request.form['image_url']
        product_name = request.form['product_name']
        total_quantity = request.form['total_quantity']
        total_price = request.form['total_price']

        cursor = db_connection.cursor()
        cursor.execute("INSERT INTO cart (image_url, product_name, quantity, price) VALUES (%s, %s, %s, %s)",
                       (image_url, product_name, total_quantity, total_price))
        db_connection.commit()
        cursor.close()
# pahila simple ma cartpage.php ma thiyo ailey cartpage.php lai cartpage1.php banako
        return redirect('/cart')


# yeta bata databaseko fetch garera cartpage2.html maprint garney


# MySQL Configuration
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'tastetales'
}

# Route to display the cart page


@app.route('/cart')
def cart():
    # Connect to MySQL database
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor(dictionary=True)

    # Fetch items from the cart table
    cursor.execute("SELECT * FROM cart")
    cart_items = cursor.fetchall()

    # Close MySQL connection
    cursor.close()
    conn.close()
# Print fetched items to the console
    print("Fetched Cart Items:", cart_items)
    # Render the cart page template with cart items
    return render_template('cartpage2.html', cart_items=cart_items)


# item del vayo vannalai
@app.route('/delete_cart_item', methods=['POST'])
def delete_cart_item():
    if request.method == 'POST':
        item_id = request.form['itemId']

        # Connect to MySQL database
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor()

        # Delete item with the given ID from the cart table
        cursor.execute("DELETE FROM cart WHERE id = %s", (item_id,))
        conn.commit()

        # Close MySQL connection
        cursor.close()
        conn.close()

        # Return a response indicating success
        return 'Item deleted successfully', 200


    # order ko lagi database connection
    # MySQL Configuration
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'tastetales'
}
# Route to handle form submission


@app.route('/submit_order', methods=['POST'])
def submit_order():
    global mysql_config
    if request.method == 'POST':
        social_title = request.form['gender']
        first_name = request.form['Fname']
        last_name = request.form['Lname']
        contact_number = request.form['contact']
        email = request.form['email']
        shipping_address = request.form['address']
        payment_option = request.form['payment']
        message = request.form['message']

        # Connect to MySQL database
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor()

        # Insert order data into the database
        cursor.execute("INSERT INTO orders (social_title, first_name, last_name, contact_number, email, shipping_address, payment_option, message) "
                       "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                       (social_title, first_name, last_name, contact_number, email, shipping_address, payment_option, message))
        conn.commit()

        # Close MySQL connection
        cursor.close()
        conn.close()
        paypal_sandbox_url = "https://www.sandbox.paypal.com/cgi-bin/webscr"
        paypal_params = {
            'cmd': '_xclick',
            # Replace with your PayPal sandbox email
            'business': 'sb-moje129178071@personal.example.com',
            'currency_code': 'USD',
            'amount': '20'  # Replace with the total amount to be paid
        }
        paypal_redirect_url = paypal_sandbox_url + '?' + \
            '&'.join([f"{key}={value}" for key,
                     value in paypal_params.items()])

        # Redirect to PayPal sandbox payment gateway
        return redirect(paypal_redirect_url)

    # admin pageko lagi retrive garna
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'tastetales'
}


@app.route('/checkcredential', methods=['POST'])
def check_credential():
    username = request.form.get('username')
    password = request.form.get('password')

    if username == 'manisha' and password == 'khadka':
        return redirect('/admin')
    else:
        return "Invalid credentials"

# Route to display the admin page


@app.route('/admin')
def admin():
    # Connect to MySQL database
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor(dictionary=True)

    # Fetch data from the orders table
    cursor.execute("SELECT * FROM orders")
    orders = cursor.fetchall()

    # Close MySQL connection
    cursor.close()
    conn.close()

    # Render the admin page template with the fetched data
    return render_template('admin.html', orders=orders)


# deletion in admin panel
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'tastetales'
}

# Route for deleting an order


@app.route('/deleteorder/<int:order_id>', methods=['POST'])
def delete_order(order_id):
    # Connect to MySQL database
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()

    # Delete the order from the database
    delete_sql = "DELETE FROM orders WHERE id = %s"
    cursor.execute(delete_sql, (order_id,))
    conn.commit()

    # Close MySQL connection
    cursor.close()
    conn.close()

    return redirect(url_for('admin'))


# update admin panellai
# MySQL Configuration
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'tastetales'
}

# Route to display the edit order page


@app.route('/editorder/<int:order_id>', methods=['GET', 'POST'])
def edit_order(order_id):
    # Connect to MySQL database
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor(dictionary=True)

    if request.method == 'GET':
        # Fetch existing order details
        cursor.execute("SELECT * FROM orders WHERE id = %s", (order_id,))
        order = cursor.fetchone()

        # Render the edit order form template with the retrieved order details
        return render_template('updateorder.html', order=order)

    elif request.method == 'POST':
        # Handle form submission for order update
        social_title = request.form['gender']
        first_name = request.form['Fname']
        last_name = request.form['Lname']
        contact_number = request.form['contact']
        email = request.form['email']
        shipping_address = request.form['address']
        payment_option = request.form['payment']
        message = request.form['message']

        # Update the order in the database
        cursor.execute("UPDATE orders SET social_title=%s, first_name=%s, last_name=%s, contact_number=%s, email=%s, shipping_address=%s, payment_option=%s, message=%s WHERE id=%s",
                       (social_title, first_name, last_name, contact_number, email, shipping_address, payment_option, message, order_id))
        conn.commit()

        # Redirect back to the admin panel after editing
        return redirect(url_for('admin'))

    # Close MySQL connection
    cursor.close()
    conn.close()


if __name__ == '__main__':
    app.run(debug=True)
