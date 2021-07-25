from flask import Flask, request, render_template, jsonify
import argparse
import requests
import os
import sys
from flask import send_from_directory

path = os.path.abspath(__file__) 
dir = os.path.dirname

src_path = dir(dir(path)) + os.sep + "utils"
sys.path.append(src_path)
import apis_tb


## ESTE ES EL EJEMPLO

#path = os.path.abspath(__file__) 
#dir = os.path.dirname

#path =dir(path) + os.sep + "flask"
 
#sys.path.append(path)
#import api
#print(api.mi_funcion)

# Mandatory
app = Flask(__name__)  # __name__ --> __main__  

# ---------- Flask functions ----------
@app.route("/")  
def home():
    """ Default path """
    #return app.send_static_file('greet.html')
    return "Waste_Classification_Detection_ML_proyect"


# localhost:6060/token_id?token_id=Y4290783D
@app.route('/token_id', methods=['GET'])                                                                                                                                                                          
def give_tokenid():
    s = request.args['token_id']
    json_file = apis_tb.read_json(fullpath=settings_json)
    if s == "Y4290783D":
        return json_file
    else:
        return "Wrong password"


settings_json = dir(dir(dir(path))) + os.sep + 'reports' + os.sep + "data_cleaned.json"


#settings_json = dir(dir(os.path.dirname(__file__))) + os.sep + 'reports' + os.sep + "waste_management_cleaned.json"

@app.route("/recibe_informacion")
def recibe_info():
    pass 

#---------- Other functions ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--x", type=str, help="the argument", required=True)
    args = vars(parser.parse_args())
    argument = args["x"]



    #argument =  input('Insert an argument to start the process')
    if argument == 'mary':

        print("---------STARTING PROCESS---------")
       #print(__file__)
        
        # Get the settings fullpath
        # \\ --> WINDOWS
        # / --> UNIX
        # Para ambos: os.sep

        settings_file = dir(path) + os.sep + "settings.json"
        #print(settings_file)
        # Load json from file
        json_readed = apis_tb.read_json(fullpath=settings_file)

    # Load variables from jsons
    SERVER_RUNNING = json_readed["server_running"]

    if SERVER_RUNNING:        
        # Load variables from jsons
        DEBUG = json_readed["debug"]
        HOST = json_readed["host"]
        PORT_NUM = json_readed["port"]
        app.run(debug=DEBUG, host=HOST, port=PORT_NUM)
    else:
        print('wrong password')

if __name__ == "__main__":
    main()