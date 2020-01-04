import os
from flask import Flask, render_template, request
from flask import send_from_directory
import jsonify
import pandas as pd
import numpy as np
import os
import pydicom
import cv2
import torch
from inference_rsna import pred
from utils import *
from apex import amp
from matplotlib import pyplot as plt
import shutil
folder = 'uploads/'
os.system('rm -r uploads/*')

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
        print('Directory Cleaned!')
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

apex = False
app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

tmp = torch.load('models/fold0_ep3.pt')

model = get_model('se_resnext101_32x4d')
model.load_state_dict(tmp['model'])

criterion = nn.BCEWithLogitsLoss()
optim = Adam(model.parameters(), lr=1e-3)
if apex == True:
    amp.initialize(model, optim, opt_level='O1')

# call model to predict an image
def api(full_path):
    name = full_path.split('.')[0].split('/')[-1]
    return pred(full_path, model) 

# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)
        DCM = pydicom.dcmread(full_name)
        ds = DCM.pixel_array
        save_file_name = full_name.split('.')[0]+'.png'
        convert_to_png(full_name, save_file_name)
        convert_to_png(full_name, save_file_name, 'brain')
        convert_to_png(full_name, save_file_name, 'subdural')
        convert_to_png(full_name, save_file_name, 'bone')
        indices = {0:'any', 1: 'Epidural', 2: 'Intraparenchymal', 3: 'Intraventricular', 4: 'Subarachnoid', 5:'Subdural'}
        result = api(full_name)
        print('*'*20)
        print(result)
        print('*'*20)
        accuracy = [round(i*100) for i in result]
        label = list(indices.values())
        print(label)
        heatmap_file_list = ['{}_grad_cam_{}.png'.format(save_file_name.split('.')[0], indices[cls_idx]) for cls_idx in range(1, 6)]
        heatmap_file_list = [p.split('/')[-1] for p in heatmap_file_list]
        window_file_list = [save_file_name.split('.')[0]+'_brain.png', save_file_name.split('.')[0]+'_subdural.png', save_file_name.split('.')[0]+'_bone.png']
        window_file_list = [w.split('/')[-1] for w in window_file_list]

    return render_template('predict.html', image_file_name = save_file_name.split('/')[-1], label = label, accuracy = accuracy, window_file_list=window_file_list, heatmap_file_list=heatmap_file_list)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/window_choice', methods=['GET', 'POST'])
def window_choice():
    if request.method == "POST":
        window_name = request.form.get("Windowing", None)
        if window_name != None:
            return render_template("predict.html", image_file_name = save_file_name.split('/')[-1], window_name = window_name)
    return render_template("predict.html")


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True, host= '0.0.0.0', port=9999)
    app.debug = True
