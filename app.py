from flask import Flask, jsonify
from flask_cors import cross_origin

app = Flask(__name__)

@app.route('/data')
@cross_origin(origin='*')
def get_data():
    data = {'name': 'dansel'}
    return jsonify(data)

if __name__ == '__main__':
    app.run()