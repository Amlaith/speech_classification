from flask import Flask, request, render_template, flash, redirect, jsonify
from render_answer import render_answer
from classifier import decode_command
import os
import uuid
from pydub import AudioSegment


UPLOAD_FOLDER = '..\\data\\audio\\to_process'
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
        file_name = str(uuid.uuid4()) + ".wav"
        full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(full_file_name)

        command = decode_command()
        server_response = render_answer(command)
        return server_response
    else:
        server_response = ['<p><p>']
    return render_template('index.html', serverResponse=server_response)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     # check if the post request has the file part
#     if 'file' in request.files:
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         file_name = str(uuid.uuid4()) + ".mp3"
#         full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
#         file.save(full_file_name)

#         # command = decode_command()
#         # answer = render_answer(command)
#         answer = 'answer'
#     else: answer = 'init'
#     return render_template('index.html', answer=answer)

# @app.route('/save-record', methods=['POST'])
# def save_record():
#     # check if the post request has the file part
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
#     file = request.files['file']
#     # if user does not select file, browser also
#     # submit an empty part without filename
#     if file.filename == '':
#         flash('No selected file')
#         return redirect(request.url)
#     file_name = str(uuid.uuid4()) + ".mp3"
#     full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
#     file.save(full_file_name)

#     return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True)    