from flask import Flask, jsonify, request, json, send_file
from flask_cors import CORS
import pandas as pd
import pytesseract
import openai
import mysql.connector
from io import BytesIO
from PIL import Image  
from serpapi import GoogleSearch
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
from math import radians, sin, cos, sqrt, atan2
import tensorflow as tf
import requests
import tensorflow_hub as hub

import numpy as np

model_url = 'https://www.kaggle.com/models/google/landmarks/frameworks/TensorFlow1/variations/classifier-asia-v1/versions/1'
labels_path = 'landmarks_classifier_asia_V1_label_map.csv'
API_KEY = "f3fb48eee276708e3fd68a637ee86b63"
panda = pd.read_csv(labels_path)
labels = dict(zip(panda.id, panda.name))
img_shape = (321, 321)
classifier = tf.keras.Sequential([hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")])

app = Flask(__name__)
app.secret_key = b'\xe4\xad\x0c(!<\x1dhs\xcdG\xbc\xc8\x8c\xe3\xc8\xf1\xe2\xeb\x0f#\xac5\xab'
ocr_text = ""
image_url = '' 
df = pd.read_csv('locations.csv', sep=',')
geolocator = Nominatim(user_agent="Trips")

# Clustering using KMeans
coordinates = df[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Explicitly set n_init
df['loc_clusters'] = kmeans.fit_predict(coordinates)

# Function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_core(filename):
    global ocr_text
    ocr_text = pytesseract.image_to_string(Image.open(filename))
    return ocr_text

def get_location_info(location_name):
    location_data = df[df['location'] == location_name]
    if not location_data.empty:
        latitude = location_data['latitude'].values[0]
        longitude = location_data['longitude'].values[0]
        return latitude, longitude
    else:
        return None

def get_weather(latitude, longitude):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': latitude,
        'lon': longitude,
        'appid': API_KEY,
        'units': 'metric'
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
    else:
        print(f"Error: Unable to fetch weather data. Status code: {response.status_code}")
        return None
    
def get_coordinates(location_name):
    row = df[df['location'] == location_name]
    if not row.empty:
        latitude = row['latitude'].values[0]
        longitude = row['longitude'].values[0]
        return latitude, longitude
    else:
        print(f"Location not found: {location_name}")
        return None


@app.route('/upload_image_landmark', methods=['POST'])
def upload_image_landmark():
    try:
        image_file = request.files['image']

        # Read image using PIL
        img = Image.open(image_file)
        img = img.resize(img_shape)
        img = np.array(img) / 255.0
        img = img[np.newaxis]

        results = classifier.predict(img)

        predicted_label = labels[np.argmax(results)]

        geolocator = Nominatim(user_agent="your_app_name")
        location = geolocator.geocode(predicted_label)

        if location:
            
            location_details = location.address.split(', ')
            city = location_details[-3]
            params = {
            "engine": "google_images",
            "ijn": "0",
            "q": "View" + city + "Malaysia", 
            "num": 1,
            "google_domain": "google.com.sg",
            "hl": "en",
            "gl": "sg",
            "tbs": "itp:photos,isz:l",
            "location_requested":"United States",
            "location_used":"United States",
            "api_key": "b721dfd3af3af630204858af235c804d08d9a4b56bf40f4761cb7dc71b13ea89"
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            if "images_results" in results:
                image_url = results["images_results"][0]["original"]

            input_city = city
            coordinates = get_coordinates(input_city)
            if coordinates:
                latitude, longitude = coordinates
                weather_data = get_weather(latitude, longitude)

            temperature = weather_data['main']['temp']
            description = weather_data['weather'][0]['description']

            
            input_cluster = df.loc[df['location'] == input_city, 'loc_clusters'].values[0]
            df['distance'] = df.apply(lambda row: haversine(df.loc[df['location'] == input_city, 'latitude'].values[0],
                                                        df.loc[df['location'] == input_city, 'longitude'].values[0],
                                                        row['latitude'], row['longitude']), axis=1)

            # Get the top 5 nearest places
            nearest_places = df[df['loc_clusters'] == input_cluster].sort_values(by='distance').head(6)['location'].tolist()[1:]
            place1 = nearest_places[0]
            place2 = nearest_places[1]
            place3 = nearest_places[2]
            place4 = nearest_places[3]
            place5 = nearest_places[4]
            return jsonify({
                'message': 'Image uploaded and processed successfully',
                'text': city,
                'imageurl': image_url,
                'place1': place1,
                'place2': place2,
                'place3': place3,
                'place4': place4,
                'place5': place5,
                'temperature': temperature,
                'description': description,

            })
            
        else:
            print("Location not found.")
            return jsonify({'error': 'Location not found'}), 404

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    
    try:
        image_file = request.files['image'] 
        if not image_file:
            return jsonify({'message': 'No image uploaded'}), 400

        print("Received image file:", image_file.filename)  
        global ocr_text  
        ocr_text = ocr_core(image_file)
        params = {
        "engine": "google_images",
        "ijn": "0",
        "q": "View" + ocr_text + "Malaysia", 
        "num": 1,
        "google_domain": "google.com.sg",
        "hl": "en",
        "gl": "sg",
        "tbs": "itp:photos,isz:l",
        "location_requested":"United States",
        "location_used":"United States",
        "api_key": "b721dfd3af3af630204858af235c804d08d9a4b56bf40f4761cb7dc71b13ea89"
        }

        input_city = ocr_text.strip()
        coordinates = get_coordinates(input_city)
        if coordinates:
            latitude, longitude = coordinates
            weather_data = get_weather(latitude, longitude)
        
        temperature = weather_data['main']['temp']
        description = weather_data['weather'][0]['description']

        

        # Check if the input_city exists in the DataFrame
        if input_city not in df['location'].values:
            print(f'The city "{input_city}" is not in the database')  # Debugging
            return jsonify({'message': f'The city "{input_city}" is not in the database'}), 400

        # Find the cluster of the input city
        input_cluster = df.loc[df['location'] == input_city, 'loc_clusters'].values[0]

        # Calculate distances to all cities in the same cluster
        df['distance'] = df.apply(lambda row: haversine(df.loc[df['location'] == input_city, 'latitude'].values[0],
                                                        df.loc[df['location'] == input_city, 'longitude'].values[0],
                                                        row['latitude'], row['longitude']), axis=1)

        # Get the top 5 nearest places
        nearest_places = df[df['loc_clusters'] == input_cluster].sort_values(by='distance').head(6)['location'].tolist()[1:]
        place1 = nearest_places[0]
        place2 = nearest_places[1]
        place3 = nearest_places[2]
        place4 = nearest_places[3]
        place5 = nearest_places[4]
        search = GoogleSearch(params)
        results = search.get_dict()

        if "images_results" in results:
            image_url = results["images_results"][0]["original"]

        print("ocr image url")
        print(image_url)

        return jsonify({
                'message': 'Image uploaded and processed successfully',
                'text': ocr_text,
                'imageurl': image_url,
                'place1': place1,
                'place2': place2,
                'place3': place3,
                'place4': place4,
                'place5': place5,
                'temperature': temperature,
                'description': description,
            })
    except Exception as e:
        print(f"Error processing image: {str(e)}")  # Print the error for debugging
        return jsonify({'message': f'Error processing image: {str(e)}'}), 500

@app.route('/output', methods=['POST'])
def output():
    if request.method == 'POST':
        category = request.form.get('category')
        ocr_text = request.form.get('ocrText')


        if category == '1':
            openai.api_key = "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"

            prompt = "List of place to eat and the address in" + ocr_text + "Malaysia"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides information about places to eat in the"+ ocr_text},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.05,
                )

            answer = response.choices[0].message['content']
            return jsonify({'message': 'Data sent sucesfully','answer': answer})
        
        elif category == '2':
            openai.api_key = "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"

            prompt = "History of" + ocr_text + "Malaysia"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides information about the history of the "+ ocr_text},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.05,
                )

            answer = response.choices[0].message['content']
            return jsonify({'message': 'Data sent sucesfully','answer': answer})
        
        elif category == '3':
            openai.api_key = "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"

            prompt = "List of interesting places and the address in" + ocr_text + "Malaysia"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides information about the interesting places to visit in the" + ocr_text},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.05,
                )

            answer = response.choices[0].message['content']
            return jsonify({'message': 'Data sent sucesfully','answer': answer})
        
@app.route('/place', methods=['POST'])
def place():

    if request.method == 'POST':
        data = request.get_json()
        place = data.get('value')

        params = {
        "engine": "google_images",
        "ijn": "0",
        "q": "View" + place + "Malaysia", 
        "num": 1,
        "google_domain": "google.com.sg",
        "hl": "en",
        "gl": "sg",
        "tbs": "itp:photos,isz:l",
        "location_requested":"United States",
        "location_used":"United States",
        "api_key": "b721dfd3af3af630204858af235c804d08d9a4b56bf40f4761cb7dc71b13ea89"
        }

        input_city = place
        coordinates = get_coordinates(input_city)
        if coordinates:
            latitude, longitude = coordinates
            weather_data = get_weather(latitude, longitude)
        
        temperature = weather_data['main']['temp']
        description = weather_data['weather'][0]['description']
        print("/place teemp")
        print(temperature)
        print("/description")
        print(description)
        
        if input_city not in df['location'].values:
            print(f'The city "{input_city}" is not in the database')  # Debugging
            return jsonify({'message': f'The city "{input_city}" is not in the database'}), 400

        # Find the cluster of the input city
        input_cluster = df.loc[df['location'] == input_city, 'loc_clusters'].values[0]

        # Calculate distances to all cities in the same cluster
        df['distance'] = df.apply(lambda row: haversine(df.loc[df['location'] == input_city, 'latitude'].values[0],
                                                        df.loc[df['location'] == input_city, 'longitude'].values[0],
                                                        row['latitude'], row['longitude']), axis=1)

        # Get the top 5 nearest places
        nearest_places = df[df['loc_clusters'] == input_cluster].sort_values(by='distance').head(6)['location'].tolist()[1:]
        place1 = nearest_places[0]
        place2 = nearest_places[1]
        place3 = nearest_places[2]
        place4 = nearest_places[3]
        place5 = nearest_places[4]
        search = GoogleSearch(params)
        results = search.get_dict()

        if "images_results" in results:
            image_url = results["images_results"][0]["original"]
        print(place)
        print(place1)
        print(image_url)
    return jsonify({
                'message': 'Image uploaded and processed successfully',
                'text': place,
                'imageurl': image_url,
                'place1': place1,
                'place2': place2,
                'place3': place3,
                'place4': place4,
                'place5': place5,
                'temperature': temperature,
                'description': description,
            })
        
        

if __name__ == "__main__":
    app.run(host='localhost', port=5000)