#!/bin/bash

pip install casadi
mkdir -p /tmp/python-packages/ 
cd /tmp/python-packages/ 
git clone https://github.com/abcamiletto/urdf2casadi-light.git
cd urdf2casadi-light
pip install .
cd ..
git clone https://github.com/abcamiletto/urdf_optcontrol.git
cd urdf_optcontrol
pip install .
