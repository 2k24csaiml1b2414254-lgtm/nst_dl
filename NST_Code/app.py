import os
import torch
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VGG_PATH     = os.path.join(BASE_DIR, 'vgg_normalised.pth')
DECODER_PATH = os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_final.pth')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder(VGG_PATH).to(DEVICE)
encoder.eval()

decoder = Decoder().to(DEVICE)
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
decoder.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, alpha=1.0):
    c = transform(content_image).unsqueeze(0).to(DEVICE)
    s = transform(style_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        c_feats = encoder(c, is_test=True)
        s_feats = encoder(s, is_test=True)

        t = adaptive_instance_normalization(c_feats, s_feats)
        t = alpha * t + (1 - alpha) * c_feats

        output = decoder(t).clamp(0, 1)

    output = output.squeeze(0).cpu()
    output = transforms.ToPILImage()(output)
    return output


@app.route('/', methods=['GET', 'POST'])
def index():
    result_image    = None
    content_filename = None
    style_filename   = None
    error            = None
    alpha            = 1.0

    if request.method == 'POST':
        alpha = float(request.form.get('alpha', 1.0))

        # handle content image
        if 'content' in request.files and request.files['content'].filename:
            f = request.files['content']
            if allowed_file(f.filename):
                content_filename = secure_filename(f.filename)
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
        else:
            content_filename = request.form.get('content_path')

        # handle style image
        if 'style' in request.files and request.files['style'].filename:
            f = request.files['style']
            if allowed_file(f.filename):
                style_filename = secure_filename(f.filename)
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
        else:
            style_filename = request.form.get('style_path')

        if content_filename and style_filename:
            try:
                content_img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], content_filename)).convert('RGB')
                style_img   = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], style_filename)).convert('RGB')

                result = style_transfer(content_img, style_img, alpha)

                result_filename = 'stylized_' + content_filename
                result.save(os.path.join(app.config['UPLOAD_FOLDER'], result_filename))
                result_image = result_filename
            except Exception as e:
                error = str(e)
        else:
            error = 'Please upload both content and style images.'

    return render_template('index.html',
                           result_image=result_image,
                           content_image=content_filename,
                           style_image=style_filename,
                           alpha=alpha,
                           error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    app.run(debug=True)
