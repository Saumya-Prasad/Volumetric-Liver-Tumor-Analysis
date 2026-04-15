@echo off
set "PYTHON_EXE=C:\Users\Saumya_Shambhavi\anaconda3\envs\liver_tumor\python.exe"

echo Running Conv AE...
%PYTHON_EXE% train.py --model conv_ae --epochs 5 --batch 16

echo Running AE Flow...
%PYTHON_EXE% train.py --model ae_flow --epochs 5 --batch 16

echo Running Masked AE...
%PYTHON_EXE% train.py --model masked_ae --epochs 5 --batch 8

echo Running CCB AAE...
%PYTHON_EXE% train.py --model ccb_aae --epochs 5 --batch 16

echo Running QFormer...
%PYTHON_EXE% train.py --model qformer --epochs 5 --batch 8

echo Running Ensemble...
%PYTHON_EXE% train.py --model ensemble --epochs 5 --batch 16

echo All models finished!
