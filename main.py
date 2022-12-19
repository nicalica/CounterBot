from flask import Flask, request, render_template
from models import DialoGPT, ResponseGenerator, ToxicDetector


app = Flask(__name__)

detector = ToxicDetector()
counter_generator = ResponseGenerator()
normal_generator = DialoGPT()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get')
def get_response():
    text = request.args.get('msg')
    label = detector.classify(text)

    if label == "hate":
        response = counter_generator.generate(text)
    else:
        response = normal_generator.generate(text)

    return response


if __name__ == '__main__':
    app.run()
