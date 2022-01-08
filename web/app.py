from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
import random
import colorsys
import os.path
import nncolor
import nncolor.data as ncd

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


def next_color_pair():
    circle_rgb = next_circle_rgb()
    bg_rgb = next_bg_rgb()
    while ncd.is_neither_with_high_confidence(circle_rgb, bg_rgb):
        circle_rgb = next_circle_rgb()
        bg_rgb = next_bg_rgb()
    return (circle_rgb, bg_rgb)


@app.route('/_add_result_v2', methods=['POST'])
def add_result_v2():
    """Add a data entry and return a new stimulus.

    v2 differs from v1 by avoiding showing a stimulus for which we have a 
    high confidence that it will be classed as "neither". 
    """
    result = []
    data = request.get_json()
    print(data)
    circle_rgb = data['circle_rgb']
    bg_rgb = data['bg_rgb']
    ans = data['ans']
    add_header = not os.path.isfile(OUT_FILE)
    s = pd.DataFrame([{'ans':ans, 
        'circle_r':circle_rgb[0], 
        'circle_g':circle_rgb[1], 
        'circle_b':circle_rgb[2], 
        'bg_r':bg_rgb[0],
        'bg_g':bg_rgb[1],
        'bg_b':bg_rgb[2]}])
    s.to_csv(OUT_FILE, mode='a', index=False, header=add_header)
    next_c, next_bg = next_color_pair()
    next_colors = {'circle_rgb': next_c, 'bg_rgb': next_bg}
    return jsonify(next_colors)


@app.route('/_add_result', methods=['POST'])
def add_result():
    result = []
    data = request.get_json()
    print(data)
    circle_rgb = data['circle_rgb']
    bg_rgb = data['bg_rgb']
    ans = data['ans']
    add_header = not os.path.isfile(OUT_FILE)
    s = pd.DataFrame([{'ans':ans, 
        'circle_r':circle_rgb[0], 
        'circle_g':circle_rgb[1], 
        'circle_b':circle_rgb[2], 
        'bg_r':bg_rgb[0],
        'bg_g':bg_rgb[1],
        'bg_b':bg_rgb[2]}])
    s.to_csv(OUT_FILE, mode='a', index=False, header=add_header)
    next_colors = {
            'circle_rgb': next_circle_rgb(), 
            'bg_rgb': next_bg_rgb()}
    return jsonify(next_colors)


if __name__ == '__main__':
    app.run()
