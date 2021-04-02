from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
import random
import colorsys
import os.path

OUT_FILE = './out/results.csv'
HUE_RANGE = [0.04, 0.177]

app = Flask("Color space experiments")


@app.route('/')
def index():
    return render_template('index.html', title='Experiment 1')


def next_circle_rgb():
    hsv = [random.uniform(*HUE_RANGE), 
            random.random(),
            random.random()]
    rgb = colorsys.hsv_to_rgb(*hsv)
    return rgb


def next_bg_rgb():
    grey = random.random()
    return [grey, grey, grey]


@app.route('/_add_result', methods=['POST'])
def add_result():
    result = []
    data = request.get_json()
    print(data)
    circle_rgb = data['circle_rgb']
    bg_rgb = data['bg_rgb']
    ans = data['ans']
    add_header = not os.path.isfile(OUT_FILE)
    s = pd.DataFrame([{'ans':ans, 'circle_rgb':circle_rgb, 'bg_rgb':bg_rgb}])
    s.to_csv(OUT_FILE, mode='a', index=False, header=add_header)
    next_colors = {
            'circle_rgb': next_circle_rgb(), 
            'bg_rgb': next_bg_rgb()}
    return jsonify(next_colors)


if __name__ == '__main__':
    app.run()
