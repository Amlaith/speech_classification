from flask import Flask, request, render_template
from render_answer import render_answer
from classifier import decode_command


app = Flask(__name__)

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
        answer = 'Напишите что-нибудь в форму'
    else:
        command = decode_command(user_input)
        answer = render_answer(command)

    # Render the template with the corresponding text
    return render_template('index.html', answer=answer)

# Create a template for the form
# @app.template_filter('render_answer')
# def render_answer(answer):
#     if answer is not None:
#         return answer
#     else:
#         return ''

if __name__ == '__main__':
    app.run(debug=True)    