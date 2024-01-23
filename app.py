from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from age_detection_code import get_permission_and_age, preprocess_face_for_age_model, postprocess_age_prediction, class_labels_reassign
from age_detection_code import face_cascade
from age_detection_code import age_model
import mysql.connector

#recommendation ko lagi import
from flask import jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler


# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')

# Load the pre-trained age detection model
model_path = "age_model_checkpoint.h5"
age_model = load_model(model_path)

# Function to process age detection and return age prediction
# app.py



@app.route('/detect_age', methods=['POST'])
def detect_age():
    permission_result = request.form.get('permission')
    if permission_result == 'yes':
        # Use the modified get_permission_and_age function
        age = get_permission_and_age(age_model, face_cascade)

        
        if age == -1:
    # Redirect to a page indicating no permission
            return "Permission denied! You cannot access this feature."

# Extract the last number from the age range string
        last_age_number = int(age.split('-')[-1])

        if last_age_number > 18:
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
    'auth_plugin': 'mysql_native_password'  # Use 'mysql_native_password' for MySQL 8
}

@app.route('/addToCart', methods=['POST'])
def addToCart():
    if request.method == 'POST':
        # Access form data using request.form
        image_url = request.form['image_url']
        product_name = request.form['product_name']
        total_quantity = int(request.form['total_quantity'])  # Ensure it's an integer
        total_price = float(request.form['total_price'])  # Ensure it's a float

        # Connect to the MySQL database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Insert data into the "cart" table
        sql = "INSERT INTO cart (image_url, product_name, quantity, price) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (image_url, product_name, total_quantity, total_price))

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

    conn = pymysql.connect(host=servername, user=username, password=password, database=dbname)

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



# @app.route('/detect_age', methods=['POST'])
# def detect_age():
#     age = get_permission_and_age()

#     if age == -1:
#         # Redirect to a page indicating no permission
#         return "Permission denied! You cannot access this feature."

#     # Extract the last number from the age range string
#     last_age_number = int(age.split('-')[-1])

#     if last_age_number > 18:
#         return redirect(url_for('harddrinks'))
#     else:
#         return redirect(url_for('underage'))
















@app.route('/permission')
def permission():
    return render_template('permission.html')

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')


# @app.route('/detect_age', methods=['POST'])
# def age_detection():
#     age = detect_age()
#     if age == -1:
#         return "Permission denied! You cannot access this feature."
#     elif age > 18:
#         return redirect(url_for('harddrinks'))
#     else:
#         return redirect(url_for('underage'))

@app.route('/harddrinks')
def harddrinks():
    return render_template('harddrinks.html')

@app.route('/underage')
def underage():
    return render_template('underage.html')










# Load your DataFrame
df1 = pd.read_excel(r'C:\project\age-integrate-pachiko-tastetales\Nepal_cheers_Liquor_Online_ALcololic_Beveragee.xlsx')

# TF-IDF vectorizer for name recommendations
tfidf_vectorizer_name = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=0.05, max_features=500)
name_data = df1['Name']
tfidf_matrix_name = tfidf_vectorizer_name.fit_transform(name_data)
cosine_similarities_name = linear_kernel(tfidf_matrix_name, tfidf_matrix_name)

# TF-IDF vectorizer for brand recommendations
brand_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
brand_matrix = brand_vectorizer.fit_transform(df1['Brand'])
brand_cosine_similarities = linear_kernel(brand_matrix, brand_matrix)

# Numeric features matrix for price recommendations
numeric_features = df1[['Price', 'Alcohol']]
scaler = MinMaxScaler()
numeric_features_normalized = scaler.fit_transform(numeric_features)
combined_matrix_price = np.concatenate([cosine_similarities_name, numeric_features_normalized], axis=1)





# @app.route('/get_name_recommendations', methods=['POST'])
# def get_name_recommendations():
#     user_input_name = request.json['name']
#     input_index = df1[df1['Name'] == user_input_name.lower()].index[0]
#     name_similarity_scores = list(enumerate(cosine_similarities_name[input_index]))
#     sorted_names = sorted(name_similarity_scores, key=lambda x: x[1], reverse=True)
#     top5_recommendations = [df1['Name'][i[0]] for i in sorted_names[1:6]]
#     return jsonify(top5_recommendations)




@app.route('/get_name_recommendations', methods=['POST'])
def get_name_recommendations():
    user_input_name = request.json['name']
    print(f"User Input Name: {user_input_name}")

    # Ensure the 'Name' column is converted to lowercase before comparison
    input_index = df1.index[df1['Name'].str.lower() == user_input_name.lower()].tolist()

    if not input_index:
        print(f"Name not found: {user_input_name}")
        return jsonify([])

    input_index = input_index[0]
    print(f"Input Index: {input_index}")

    name_similarity_scores = list(enumerate(cosine_similarities_name[input_index]))
    sorted_names = sorted(name_similarity_scores, key=lambda x: x[1], reverse=True)
    top5_recommendations = [df1['Name'][i[0]] for i in sorted_names[1:6]]

    print(f"Top 5 Recommendations: {top5_recommendations}")

    return jsonify(top5_recommendations)


# @app.route('/get_brand_recommendations', methods=['POST'])
# def get_brand_recommendations():
#     user_input_brand = request.json['brand']
#     user_brand_index = df1.index[df1['Brand'] == user_input_brand.lower()].tolist()[0]
#     brand_similarities = combined_matrix_price[user_brand_index]
#     similar_brands_indices = brand_similarities.argsort()[-6:-1][::-1]
#     similar_brands = df1.iloc[similar_brands_indices]['Brand']
#     recommended_products = df1.iloc[similar_brands_indices]['Name']
#     return jsonify(list(recommended_products))


@app.route('/get_brand_recommendations', methods=['POST'])
def get_brand_recommendations():
    user_input_brand = request.json['brand']
    print(f"User Input Brand: {user_input_brand}")

    # Ensure the 'Brand' column is converted to lowercase before comparison
    user_brand_index = df1.index[df1['Brand'].str.lower() == user_input_brand.lower()].tolist()

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





# @app.route('/get_price_recommendations', methods=['POST'])
# def get_price_recommendations():
#     user_input_price = request.json['price']
#     index = df1[df1['Price'] == int(user_input_price)].index[0]
#     similarity_scores = list(enumerate(combined_matrix_price[index]))
#     similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
#     similar_products_indices = [score[0] for score in similarity_scores[1:6]]
#     return jsonify(list(df1['Name'].iloc[similar_products_indices]))


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
    similar_products = df1[df1['Price'] >= user_input_price].sort_values(by='Price').head(5)

    if similar_products.empty:
        print("No similar products found.")
        return jsonify([])

    recommended_products = list(similar_products['Name'])
    
    print(f"Recommended Products: {recommended_products}")

    return jsonify(recommended_products)


# # Load your DataFrame
# df1 = pd.read_excel(r'C:\project\age-integrate-pachiko-tastetales\Nepal_cheers_Liquor_Online_ALcololic_Beveragee.xlsx')



# # TF-IDF vectorizer for name recommendations
# tfidf_vectorizer_name = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=0.05, max_features=500)
# name_data = df1['Name']
# tfidf_matrix_name = tfidf_vectorizer_name.fit_transform(name_data)
# cosine_similarities_name = linear_kernel(tfidf_matrix_name, tfidf_matrix_name)

# # TF-IDF vectorizer for brand recommendations
# brand_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
# brand_matrix = brand_vectorizer.fit_transform(df1['Brand'])
# brand_cosine_similarities = linear_kernel(brand_matrix, brand_matrix)

# # Numeric features matrix for price recommendations
# numeric_features = df1[['Price', 'Alcohol']]
# scaler = MinMaxScaler()
# numeric_features_normalized = scaler.fit_transform(numeric_features)
# combined_matrix_price = np.concatenate([cosine_similarities_name, numeric_features_normalized], axis=1)



# @app.route('/get_name_recommendations', methods=['POST'])
# def get_name_recommendations():
#     user_input_name = request.json['name']
#     input_index = df1[df1['Name'] == user_input_name.lower()].index[0]
#     name_similarity_scores = list(enumerate(cosine_similarities_name[input_index]))
#     sorted_names = sorted(name_similarity_scores, key=lambda x: x[1], reverse=True)
#     top5_recommendations = [df1['Name'][i[0]] for i in sorted_names[1:6]]
#     return jsonify(top5_recommendations)

# @app.route('/get_brand_recommendations', methods=['POST'])
# def get_brand_recommendations():
#     user_input_brand = request.json['brand']
#     user_brand_index = df1.index[df1['Brand'] == user_input_brand.lower()].tolist()[0]
#     brand_similarities = combined_matrix_price[user_brand_index]
#     similar_brands_indices = brand_similarities.argsort()[-6:-1][::-1]
#     similar_brands = df1.iloc[similar_brands_indices]['Brand']
#     recommended_products = df1.iloc[similar_brands_indices]['Name']
#     return jsonify(list(recommended_products))

# @app.route('/get_price_recommendations', methods=['POST'])
# def get_price_recommendations():
#     user_input_price = request.json['price']
#     index = df1[df1['Price'] == int(user_input_price)].index[0]
#     similarity_scores = list(enumerate(combined_matrix_price[index]))
#     similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
#     similar_products_indices = [score[0] for score in similarity_scores[1:6]]
#     return jsonify(list(df1['Name'].iloc[similar_products_indices]))







if __name__ == '__main__':
    app.run(debug=True)

