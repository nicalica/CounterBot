from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
# from simpleclassifier import target_classifier
from perspective_client import perspective_client
# from CounterSpeechGenerator import GPT2Generator

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
#
# init_target_classifier = target_classifier()
# init_GPT2_generator = GPT2Generator()
init_perspective_client = perspective_client()


perpsective_chat_bot_conversations = []


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'


@app.route('/Perspective_analysis', methods=['POST', 'GET'])
def Perspective_analysis():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            perpsective_chat_bot_conversations.append('YOU: ' + sent)
            analyse_result = init_perspective_client.get_toxicity(sent)
            formatted_result = init_perspective_client.print_formatted_analysis(analyse_result)
            perpsective_chat_bot_conversations.append('BOT: ' + formatted_result)
    return render_template('Perspective_analysis.html', conversations=perpsective_chat_bot_conversations)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def main():
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
