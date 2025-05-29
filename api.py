from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
from ml_model import predict_go_terms, load_model
from graph_model import refine_predictions, load_go_data

app = Flask(__name__)
CORS(app)  # Enable CORS untuk frontend

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Memuat model dan graf GO
print("Memuat model dan graf GO...")
try:
    model, label_info = load_model('dna_annotation_model.pkl')
    go_terms, relationships, G = load_go_data()
    model_loaded = True
except Exception as e:
    print(f"Error memuat model: {e}")
    model_loaded = False

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk memprediksi anotasi GO dari urutan DNA
    """
    if not model_loaded:
        return jsonify({
            'error': 'Model belum dimuat. Silakan latih model terlebih dahulu.'
        }), 400
    
    data = request.json
    
    if 'sequence' not in data:
        return jsonify({
            'error': 'Urutan DNA tidak ditemukan dalam request'
        }), 400
    
    sequence = data['sequence']
    
    # Validasi urutan DNA
    valid_bases = set('ATGC')
    if not all(base in valid_bases for base in sequence.upper()):
        return jsonify({
            'error': 'Urutan DNA tidak valid. Hanya basa A, T, G, dan C yang diperbolehkan.'
        }), 400
    
    # Memprediksi anotasi GO
    predictions = predict_go_terms(sequence, model, label_info)
    
    # Memperbaiki prediksi berdasarkan struktur GO
    refined_predictions = refine_predictions(predictions, G)
    
    return jsonify({
        'predictions': predictions,
        'refined_predictions': refined_predictions
    })

@app.route('/train', methods=['POST'])
def train():
    """
    Endpoint untuk melatih model
    """
    from ml_model import load_and_process_data, train_model, evaluate_model, save_model
    
    try:
        # Memuat dan memproses data
        X_train, X_test, y_train, y_test, label_info, data = load_and_process_data()
        
        # Melatih model
        model = train_model(X_train, y_train)
        
        # Mengevaluasi model
        eval_results = evaluate_model(model, X_test, y_test, label_info)
        
        # Menyimpan model
        save_model(model, label_info)
        
        return jsonify({
            'status': 'success',
            'message': 'Model berhasil dilatih',
            'evaluation': eval_results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error saat melatih model: {str(e)}'
        }), 500

@app.route('/upload', methods=['POST'])
def upload():
    """
    Endpoint untuk mengunggah dataset
    """
    if 'gene_seq' not in request.files or 'gene_go' not in request.files or 'gene_locus' not in request.files:
        return jsonify({
            'error': 'File dataset tidak lengkap'
        }), 400
    
    try:
        gene_seq = request.files['gene_seq']
        gene_go = request.files['gene_go']
        gene_locus = request.files['gene_locus']
        
        gene_seq.save('gene_seq.csv')
        gene_go.save('gene_data_GO.csv')
        gene_locus.save('gene_data_locus.csv')
        
        return jsonify({
            'status': 'success',
            'message': 'Dataset berhasil diunggah'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error saat mengunggah dataset: {str(e)}'
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """
    Endpoint untuk memeriksa status model
    """
    return jsonify({
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)