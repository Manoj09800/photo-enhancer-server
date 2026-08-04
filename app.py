import os
import io
import sys
import gc
import time
import traceback
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import torch

app = Flask(__name__)
CORS(app)

torch.set_num_threads(2)
device = torch.device('cpu')
model = None

# Keep input small so CPU inference finishes in reasonable time and low memory
MAX_INPUT_SIDE = 480
TILE_SIZE = 160
TILE_OVERLAP = 12


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


def predict_tiled(m, img, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    """Process the image in small tiles to keep peak memory low on
    memory-constrained free hosting (avoids OOM kills on large images)."""
    w, h = img.size
    scale = m.scale
    out = Image.new('RGB', (w * scale, h * scale))
    step = tile_size - overlap

    y0 = 0
    while y0 < h:
        x0 = 0
        y1 = min(y0 + tile_size, h)
        while x0 < w:
            x1 = min(x0 + tile_size, w)
            tile = img.crop((x0, y0, x1, y1))
            sr_tile = m.predict(tile)
            out.paste(sr_tile, (x0 * scale, y0 * scale))
            del tile, sr_tile
            gc.collect()
            x0 += step
        y0 += step

    return out


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
        with torch.no_grad():
            result = predict_tiled(m, img)
        print(f"Inference took {time.time() - t0:.1f}s for image size {img.size}", flush=True)

        buf = io.BytesIO()
        result.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        del result
        gc.collect()
        return send_file(buf, mimetype='image/jpeg')
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
