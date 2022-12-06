#!/bin/env python
# -*coding: UTF-8 -*-
#
# export FLASK_DEBUG=True
# export FLASK_APP=app.py
# flask run
# flask run --host=134.246.146.178 # Laptop
# flask run --host=134.246.146.54 # Pacnet
#
#

from flask import Flask

app = Flask(__name__)
app.config.from_object('config')  # app.config["VAR_NAME"]
# app = Flask(__name__, instance_relative_config=True)
# app.config.from_object('config')
# app.config.from_pyfile('config.py')
print("myapp/__init__.py:", app.config)

import myapp.views

if __name__ == '__main__':
    app.run()
