from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from age_detection_code import get_permission_and_age, preprocess_face_for_age_model, postprocess_age_prediction, class_labels_reassign
from age_detection_code import face_cascade
from age_detection_code import age_model


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



# @app.route('/detect_age', methods=['POST'])
# def detect_age():
#     permission_result = request.form.get('permission')
#     if permission_result == 'yes':
#         # Use the modified get_permission_and_age function
#         age = get_permission_and_age(age_model, face_cascade)

        
#         if age == -1:
#     # Redirect to a page indicating no permission
#             return "Permission denied! You cannot access this feature."

# # Extract the last number from the age range string
#         last_age_number = int(age.split('-')[-1])

#         if last_age_number > 18:
#             return redirect(url_for('harddrinks'))
#         else:
#             return redirect(url_for('underage'))
        
@app.route('/')
def index():
    return render_template('index.html')




@app.route('/detect_age', methods=['POST'])
def detect_age():
    age = get_permission_and_age()

    if age == -1:
        # Redirect to a page indicating no permission
        return "Permission denied! You cannot access this feature."

    # Extract the last number from the age range string
    last_age_number = int(age.split('-')[-1])

    if last_age_number > 18:
        return redirect(url_for('harddrinks'))
    else:
        return redirect(url_for('underage'))
















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





@app.route('/get_name_recommendations', methods=['POST'])
def get_name_recommendations():
    user_input_name = request.json['name']
    input_index = df1[df1['Name'] == user_input_name.lower()].index[0]
    name_similarity_scores = list(enumerate(cosine_similarities_name[input_index]))
    sorted_names = sorted(name_similarity_scores, key=lambda x: x[1], reverse=True)
    top5_recommendations = [df1['Name'][i[0]] for i in sorted_names[1:6]]
    return jsonify(top5_recommendations)


@app.route('/get_brand_recommendations', methods=['POST'])
def get_brand_recommendations():
    user_input_brand = request.json['brand']
    user_brand_index = df1.index[df1['Brand'] == user_input_brand.lower()].tolist()[0]
    brand_similarities = combined_matrix_price[user_brand_index]
    similar_brands_indices = brand_similarities.argsort()[-6:-1][::-1]
    similar_brands = df1.iloc[similar_brands_indices]['Brand']
    recommended_products = df1.iloc[similar_brands_indices]['Name']
    return jsonify(list(recommended_products))


@app.route('/get_price_recommendations', methods=['POST'])
def get_price_recommendations():
    user_input_price = request.json['price']
    index = df1[df1['Price'] == int(user_input_price)].index[0]
    similarity_scores = list(enumerate(combined_matrix_price[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_products_indices = [score[0] for score in similarity_scores[1:6]]
    return jsonify(list(df1['Name'].iloc[similar_products_indices]))












# pahiloko maybe milirako ni huna sakcha

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

