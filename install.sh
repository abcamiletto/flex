#!/bin/bash

pip install casadi
mkdir -p /tmp/python-packages/ 
cd /tmp/python-packages/ 
git clone https://github.com/abcamiletto/urdf2casadi-light.git
cd urdf2casadi-light
pip install .
cd ..
git clone https://github.com/abcamiletto/urdf2optcontrol.git
cd urdf2optcontrol
pip install .
