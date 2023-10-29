from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/sum/<int:a>/<int:b>")
def sum(a, b):
    return str(a + b)

@app.route("/model", methods=["POST"])
def model():
    js = request.get_json("var_name")
    x = js['x']
    y = js['y']
    return str(x + y)