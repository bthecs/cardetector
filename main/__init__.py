import os
from flask import Flask
from dotenv import load_dotenv

def create_app():
    app = Flask(__name__);
    load_dotenv()

    from main.controllers import detector

    app.register_blueprint(detector, url_prefix='/detector')

    return app