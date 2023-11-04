from flask import Flask, request, render_template, flash, redirect
from render_answer import render_answer
from classifier import decode_command
import os
import uuid


UPLOAD_FOLDER = '..\\data\\audio'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# d = {'one': 'that\'s one', 'two': 'this is two', 'three': 'three here'}
# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

# Create a route for the form
@app.route('/', methods=['GET', 'POST'])
def index():
    # Get the user input from the form
    user_input = (request.form.get('user_input'))

    # Get the corresponding text from the dictionary
    # answer = d.get(user_input)
    
    if user_input is None:
        answer = ''
    else:
        command = decode_command(user_input)
        answer = render_answer(command)

    # Render the template with the corresponding text
    return render_template('index.html', answer=answer)

@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    file_name = str(uuid.uuid4()) + ".mp3"
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)
    return '<h1>Success</h1>'

if __name__ == '__main__':
    app.run(debug=True)    