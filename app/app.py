from flask import Flask, request, render_template, flash, redirect, jsonify
from render_answer import render_answer
from classifier import decode_with_cnn, decode_with_tabular
from utils.spec_maker import transform_input_to_spec
import os
import uuid
# from pydub import AudioSegment


UPLOAD_FOLDER = 'to_process'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# d = {'one': 'that\'s one', 'two': 'this is two', 'three': 'three here'}
# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

# Create a route for the form
@app.route('/', methods=['GET', 'POST'])
def index():    
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        # file_name = str(uuid.uuid4()) + ".wav"
        file_name = "input_audio.wav"
        full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(full_file_name)
        # transform_input_to_spec()
        # command = decode_with_cnn()
        command = decode_with_tabular()
        server_response = render_answer(command)
        return server_response
    else:
        server_response = ['<p><p>']
    return render_template('index.html', serverResponse=server_response)


if __name__ == '__main__':
    app.run(debug=True)    