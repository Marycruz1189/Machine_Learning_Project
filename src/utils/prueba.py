import os
import sys

import requests
dir = os.path.dirname
src_path = dir(os.path.abspath(__file__))

fullpath = src_path + os.sep + 'resources' + os.sep + "imagenes.png"

print(os.path.abspath(__file__))