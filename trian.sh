#!/usr/bin/env bash
echo "CNN train started"
python3 CNN_model_train.py
echo "CNN train completed"

echo "verify model train started"
python3 verify_model_train.py
echo "verify model train completed"

echo "RNN start train"
python3 RNN_model_train.py
echo "RNN train completed"