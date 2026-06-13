#!/bin/bash

# 1. Percorsi e Ambiente
VENV_DIR="/home/scalzolaro/.venv"
source "$VENV_DIR/bin/activate"

export TF_CPP_MIN_LOG_LEVEL=3

# Silenzia i messaggi di logging interni di Abseil (usati da XLA)
export TF_CPP_MIN_VLOG_LEVEL=0

# 2. FIX CRITICO: Troviamo dove risiede il driver fisico (libcuda.so)
# Solitamente su Ubuntu è in /usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# 3. FIX LIBRERIE VENV: Agganciamo le librerie nvidia installate via pip
# Questo comando estrae automaticamente tutti i percorsi delle sottocartelle nvidia nel venv
NVIDIA_LIBS=$(python3 -c "import nvidia; import os; print(':'.join([os.path.join(os.path.dirname(nvidia.__file__), d, 'lib') for d in os.listdir(os.path.dirname(nvidia.__file__)) if os.path.isdir(os.path.join(os.path.dirname(nvidia.__file__), d, 'lib'))]))")
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"

# 4. Forziamo TensorFlow a ignorare errori minori di caricamento e a mappare la GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$VENV_DIR

echo "--- TEST GPU ---"
python3 -c "import tensorflow as tf; print('Dispositivi trovati:', tf.config.list_physical_devices()); print('GPU disponibile:', tf.test.is_gpu_available())"
echo "----------------"

# Procedi con i tuoi comandi python...
:'
echo "================================="
echo "================1================"
echo "================================="
python3 masked/NN/hp_tuning.py none fixed SBOX_OUT categorical
echo "================================="
echo "================2================"
echo "================================="
python3 masked/NN/hp_tuning.py scalar fixed SBOX_OUT categorical 
echo "================================="
echo "================3================"
echo "================================="
python3 masked/NN/hp_tuning.py binary fixed SBOX_OUT categorical 
echo "================================="
echo "================4================"
echo "================================="
python3 masked/NN/hp_tuning.py none fixed SBOX_OUT binary
echo "================================="
echo "================5================"
echo "================================="
python3 masked/NN/hp_tuning.py scalar fixed SBOX_OUT binary
'
echo "================================="
echo "================6================"
echo "================================="
python3 masked/NN/hp_tuning.py binary fixed SBOX_OUT binary 
echo "================================="
echo "================7================"
echo "================================="
python3 masked/NN/hp_tuning.py none fixed HW_SO categorical
echo "================================="
echo "================8================"
echo "================================="
python3 masked/NN/hp_tuning.py scalar fixed HW_SO categorical
echo "================================="
echo "================9================"
echo "================================="
python3 masked/NN/hp_tuning.py binary fixed HW_SO categorical   
