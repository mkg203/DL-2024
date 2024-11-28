#!/bin/bash

date >> model_logs.txt

python3.11 run.py -corpus ../data.csv -set ../PrIMuS/ -vocabulary Data/vocabulary_agnostic.txt| tee -a model_logs.txt

echo >> model_logs.txt
