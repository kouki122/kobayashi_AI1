from flask import Flask, render_template, request, jsonify, redirect, url_for
import librosa
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'files'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size: 16MB

# モデルの再構築と再保存（time_majorを除去）
def rebuild_and_save_models():
    # Pitch Model
    pitch_model = Sequential()
    pitch_model.add(LSTM(300, input_shape=(4, 1), return_sequences=False))
    pitch_model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    pitch_model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mae'])
    
    # Kerasのネイティブ形式でモデルを保存
    pitch_model.save("my_h4_model.keras")

    # Tempo Model
    tempo_model = Sequential()
    tempo_model.add(LSTM(300, input_shape=(4, 1), return_sequences=False))
    tempo_model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    tempo_model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mae'])
    
    # Kerasのネイティブ形式でモデルを保存
    tempo_model.save("my_h3_model_tempo.keras")

# 一度だけモデルを再構築して保存
rebuild_and_save_models()

# カスタムオブジェクトとしてOrthogonal初期化子を登録
custom_objects = {
    'Orthogonal': Orthogonal
}

# モデルのロード時にカスタムオブジェクトを指定
pitch_model = load_model('my_h4_model.keras', custom_objects=custom_objects)
tempo_model = load_model('my_h3_model_tempo.keras', custom_objects=custom_objects)

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
        if pitch_difference > 0: 
            pitch_result ="high"  
        else: 
            pitch_result= "low"
        results['pitch_difference'] = float(abs(pitch_difference[0][0]))
        results['pitch_result'] = pitch_result
    else:
        results['pitch_model'] = "ピッチモデルが存在しません。"

    if tempo_model:
        new_tempo_input = np.array([[avg_tempo] * 4]).reshape(-1, 4, 1)
        predicted_tempo = tempo_model.predict(new_tempo_input)
        tempo_difference = (predicted_tempo - avg_tempo) / avg_tempo * 100
        if tempo_difference > 0:
            tempo_result = "fast"
        else:
            tempo_result = "slow"
        results['tempo_difference'] = float(abs(tempo_difference[0][0]))
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

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        results = evaluate_new_audio(file_path, pitch_model, tempo_model)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(file_path)

    return jsonify(results)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
