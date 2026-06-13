#!/bin/bash

# 1. Attiva l'ambiente virtuale
VENV_DIR="/home/scalzolaro/.venv"
source "$VENV_DIR/bin/activate"

# 2. CARICAMENTO CORRETTO DELLA GPU (NVIDIA A2)
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
NVIDIA_LIBS=$(python3 -c "import nvidia; import os; print(':'.join([os.path.join(os.path.dirname(nvidia.__file__), d, 'lib') for d in os.listdir(os.path.dirname(nvidia.__file__)) if os.path.isdir(os.path.join(os.path.dirname(nvidia.__file__), d, 'lib'))]))")
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$VENV_DIR
export TF_CPP_MIN_LOG_LEVEL=3

echo "--- TEST GPU ---"
python3 -c "import tensorflow as tf; print('GPU disponibile per attacco:', tf.test.is_gpu_available())"
echo "----------------"

# 3. COMANDI DI ATTACCO (con i percorsi corretti!)
echo "================================="
echo "================1================"
echo "================================="
python3 riscure_attack.py none fixed SBOX_OUT categorical CNN_ZAID
python3 riscure_attack.py none fixed SBOX_OUT categorical MLP

echo "================================="
echo "================2================"
echo "================================="
python3 riscure_attack.py scalar fixed SBOX_OUT categorical CNN_ZAID
python3 riscure_attack.py scalar fixed SBOX_OUT categorical MLP


echo "================================="
echo "================3================"
echo "================================="
python3 riscure_attack.py binary fixed SBOX_OUT categorical CNN_ZAID
python3 riscure_attack.py binary fixed SBOX_OUT categorical MLP


echo "================================="
echo "================4================"
echo "================================="
python3 riscure_attack.py none fixed SBOX_OUT binary CNN_ZAID
python3 riscure_attack.py none fixed SBOX_OUT binary MLP

echo "================================="
echo "================5================"
echo "================================="
python3 riscure_attack.py scalar fixed SBOX_OUT binary CNN_ZAID
python3 riscure_attack.py scalar fixed SBOX_OUT binary MLP

echo "================================="
echo "================6================"
echo "================================="
python3 riscure_attack.py binary fixed SBOX_OUT binary CNN_ZAID
python3 riscure_attack.py binary fixed SBOX_OUT binary MLP


echo "================================="
echo "================7================"
echo "================================="
python3 riscure_attack.py none fixed HW_SO categorical CNN_ZAID
python3 riscure_attack.py none fixed HW_SO categorical MLP


echo "================================="
echo "================8================"
echo "================================="
python3 riscure_attack.py scalar fixed HW_SO categorical CNN_ZAID
python3 riscure_attack.py scalar fixed HW_SO categorical MLP


echo "================================="
echo "================9================"
echo "================================="
python3 riscure_attack.py binary fixed HW_SO categorical CNN_ZAID
python3 riscure_attack.py binary fixed HW_SO categorical MLP
