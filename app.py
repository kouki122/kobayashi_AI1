from flask import Flask, render_template, request, jsonify, redirect, url_for
import librosa
import numpy as np
import os
from tensorflow.keras.models import load_model
import h5py

app = Flask(__name__)

# モデルのロード
pitch_model = load_model('my_h4_model.h5')
tempo_model = load_model('my_h3_model_tempo.h5')

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return avg_pitch, tempo

def evaluate_new_audio(file_path, pitch_model=None, tempo_model=None):
    avg_pitch, avg_tempo = extract_features(file_path)
    results = {}
    if pitch_model:
        new_pitch_input = np.array([[avg_pitch] * 4]).reshape(-1, 4, 1)
        predicted_pitch = pitch_model.predict(new_pitch_input)
        pitch_difference = (predicted_pitch - avg_pitch) / avg_pitch * 100
        pitch_result = "高い" if pitch_difference > 0 else "低い"
        results['pitch_difference'] = abs(pitch_difference[0][0])
        results['pitch_result'] = pitch_result
    else:
        results['pitch_model'] = "ピッチモデルが存在しません。"

    if tempo_model:
        new_tempo_input = np.array([[avg_tempo] * 4]).reshape(-1, 4, 1)
        predicted_tempo = tempo_model.predict(new_tempo_input)
        tempo_difference = (predicted_tempo - avg_tempo) / avg_tempo * 100
        tempo_result = "速い" if tempo_difference > 0 else "遅い"
        results['tempo_difference'] = abs(tempo_difference[0][0])
        results['tempo_result'] = tempo_result
    else:
        results['tempo_model'] = "テンポモデルが存在しません。"

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "ファイルが見つかりませんでした。"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません。"}), 400

    file_path = os.path.join('files', file.filename)
    file.save(file_path)

    results = evaluate_new_audio(file_path, pitch_model, tempo_model)
    
    # 処理後にファイルを削除
    os.remove(file_path)

    return jsonify(results)

if __name__ == '__main__':
    os.makedirs('files', exist_ok=True)
    app.run(debug=True)
