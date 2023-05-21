export PYTHONPATH=$PWD
python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000
python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000