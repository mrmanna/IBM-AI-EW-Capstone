import os
import csv
from datetime import datetime

LOG_DIR = "logs"
TRAIN_LOG = os.path.join(LOG_DIR, "train.log")
PREDICT_LOG = os.path.join(LOG_DIR, "predict.log")

def update_train_log(tag, dates, metrics, runtime, model_version, model_version_note, test=False):
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    
    log_file = TRAIN_LOG if not test else TRAIN_LOG.replace(".log", "-test.log")
    
    with open(log_file, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow([datetime.now(), tag, dates, metrics, runtime, model_version, model_version_note])

def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=False):
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    
    log_file = PREDICT_LOG if not test else PREDICT_LOG.replace(".log", "-test.log")
    
    with open(log_file, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow([datetime.now(), country, y_pred, y_proba, target_date, runtime, model_version])