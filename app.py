import os
import io
import sys
import time
import traceback
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import torch

app = Flask(__name__)
CORS(app)

device = torch.device('cpu')
model = None

# Keep input small so CPU inference finishes in reasonable time on free hosting
MAX_INPUT_SIDE = 512


def load_model():
    global model
    if model is None:
        print("Loading RealESRGAN model...", flush=True)
        os.makedirs('weights', exist_ok=True)
        from RealESRGAN import RealESRGAN
        m = RealESRGAN(device, scale=2)
        m.load_weights('weights/RealESRGAN_x2.pth', download=True)
        model = m
        print("Model loaded.", flush=True)
    return model


@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "Photo Enhancer AI server is running"})


@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')

        w, h = img.size
        longest = max(w, h)
        if longest > MAX_INPUT_SIDE:
            ratio = MAX_INPUT_SIDE / longest
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        m = load_model()
        t0 = time.time()
        result = m.predict(img)
        print(f"Inference took {time.time() - t0:.1f}s for image size {img.size}", flush=True)

        buf = io.BytesIO()
        result.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
