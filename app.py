
import os
from os import listdir
import subprocess
from flask import Flask, request, render_template, send_from_directory, url_for, jsonify, redirect, Response , send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from DNN_transcript import transcript
from align import align

UPLOAD_FOLDER = 'C:\Users\Francisco A\Google Drive (facm0002@red.ujaen.es)\ULI\TFG\FlaskApp\uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app)
cors = CORS(app, resources={r"/uploads/*": {"origins": "http://localhost:5001"}, r"/download/*": {"origins": "*"}})


@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        #check if the post request has the file part
        # if ('filewav' or 'filemid') not in request.files:
        #     print('No file part')
        #     return redirect(request.url)
        filewav = request.files['filewav']
        filexml = request.files['filexml']
        # if user does not select file, browser also
        # submit an empty part without filename
        # if filewav.filename == '':
        #     print('No selected file')
        #     return redirect(request.url)

        wavname = secure_filename(filewav.filename)
        filewav.save(os.path.join(app.config['UPLOAD_FOLDER'], wavname))

        xmlname = secure_filename(filexml.filename)
        filexml.save(os.path.join(app.config['UPLOAD_FOLDER'], xmlname))

        (f, wavext) = os.path.splitext(wavname)
        (xname, xmlext) = os.path.splitext(xmlname)

        midi_file = xname+'.mid'
        if (wavext == '.wav') and (xmlext == '.xml'):
            os.chdir(app.config['UPLOAD_FOLDER'])
            res1 = subprocess.call(['C:/Program Files/MuseScore 3/bin/MuseScore3.exe','-o'+midi_file , xmlname])
            res2 = subprocess.call(['C:/Program Files/MuseScore 3/bin/MuseScore3.exe','-o'+xname+'.mpos', xmlname])
            print(res1,res2)

            Y_pred = transcript(wavname) 
            print(Y_pred.shape)

            ld = listdir(app.config['UPLOAD_FOLDER'])
            print(ld)
            p , q  = align(midi_file , Y_pred.T)
            
            p = list(p)
            q = list(q)

        return render_template('audio_TFG.html' , filename = wavname , p = p , q = q )

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename )


if __name__ == '__main__':
	# run!
	app.run(debug = True)






