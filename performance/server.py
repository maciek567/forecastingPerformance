import subprocess

from flask import Flask

app = Flask(__name__)


@app.route('/prediction', methods=['GET'])
def handle_get_request():
    run_predictions()

    response = {'status': 'success'}
    return response


def run_predictions():
    command = ['python', '../run/run_predictions.py', '--unique_ids', 'True']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()
    return_code = process.returncode
    if return_code == 0:
        print("Script executed successfully!")
    else:
        print(f"Script execution failed with return code: {return_code}")


if __name__ == '__main__':
    app.run()
