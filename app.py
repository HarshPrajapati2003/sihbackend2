from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import numpy as np
from flask_cors import CORS
from PIL import Image  # Import Image from Pillow

app = Flask(__name__)
app.debug=True
CORS(app)

model = load_model('model4.h5')


def preprocess_image(image_file):
    # Open the image using PIL (Pillow)
    img = Image.open(image_file)
    # Resize the image to the desired target size
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def my_index():
    return render_template("index.html", flask_token="Hello   world")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        list = ['Acacia Gumefora',
                'Agave Sisal',
                'Ajuga',
                'Allophylus',
                'Aloe_Ankoberenisis',
                'Aloe_Debrana',
                'Archirantus',
                'Bederjan',
                'Beresemma_Abysinica',
                'Biden',
                'Bohera',
                'Calpurnia',
                'Carissa_Spinanrum',
                'Chenopodium',
                'Chlorodundurum',
                'Climatis',
                'Clutea',
                'Cordia',
                'Crotun',
                'Devilbackbone',
                'Dovianus',
                'Eberighia',
                'Echinopes_Kebericho',
                'Ficus_Sur',
                'Hagnia_Abbysinica',
                'Haritoki',
                'Jesminium',
                'Laggeria',
                'Lemongrass',
                'Leonotes',
                'Leucas',
                'Linipia_Adonesisis',
                'Lobelia_Rehinopetanum',
                'Melitia',
                'Messa_Lanceolata',
                'Nayontara',
                'Neem',
                'Osirus',
                'Pathorkuchi',
                'Phytolleca',
                'Plantago',
                'Rumex_Abbysinica',
                'Rumix_Nervo',
                'Senecio',
                'Stephania_Abbysinica',
                'Thankuni',
                'Thymus_Schimperia',
                'Tulsi',
                'Uritica',
                'Verbasucum',
                'Vernonia_Amag',
                'Vernonia_Leop',
                'Zeneria_Scabra',
                'Zenora']
        image_file = request.files['image']
        processed_image = preprocess_image(image_file)
        predictions = model.predict(processed_image)
        # final_prediction = np.argmax(predictions)
        final_prediction = int(np.argmax(predictions))
        confidence = np.max(predictions)*100
        final_output = {
            'category': final_prediction,
            'confidence': confidence
        }
        final_output_json = json.dumps(final_output)
        # print("this is final output : ",final_output_json)
        return jsonify({'category': list[final_prediction], 'confidence': confidence})
    except Exception as e:
        print("this is ",str(e))
        return jsonify({'error': 'An error occurred during prediction'}), 500

# app.run(debug=True)