import random
import subprocess

from flask import Flask

from run.configuration import all_company_names

app = Flask(__name__)


@app.route('/prediction/arima', methods=['GET'])
def arima_prediction():
    return run_predictions("AutoArima")


@app.route('/prediction/ces', methods=['GET'])
def ces_prediction():
    return run_predictions("Ces")


@app.route('/prediction/garch', methods=['GET'])
def garch_prediction():
    return run_predictions("Garch")


@app.route('/prediction/xgboost', methods=['GET'])
def xgboost_prediction():
    return run_predictions("XGBoost")


@app.route('/prediction/reservoir', methods=['GET'])
def reservoir_prediction():
    return run_predictions("Reservoir")


@app.route('/prediction/nhits', methods=['GET'])
def nhits_prediction():
    return run_predictions("NHits")


def run_predictions(method: str):
    company_name = all_company_names[random.randint(0, 24)]
    command = ['python', '../run/run_performance.py', '--unique_ids', '--company', company_name, '--method', method]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()

    return_code = process.returncode
    if return_code == 0:
        print("Script executed successfully!")
    else:
        print(f"Script execution failed with return code: {return_code}")
    response = {'status': 'success'}
    return response
