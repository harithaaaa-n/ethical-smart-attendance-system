import traceback
from flask import Flask, request, jsonify, session
import sys, os
sys.path.append(os.getcwd())
from backend import mobile_routes
from backend import database_manager

app = Flask(__name__)
app.secret_key = 'test'
app.register_blueprint(mobile_routes.mobile_bp, url_prefix='/api/mobile')

@app.route('/test')
def run_test():
    try:
        with app.test_request_context('/api/mobile/check-attendance?user_id=117&period=5'):
            res = mobile_routes.check_attendance()
            return res.get_data(as_text=True)
    except Exception as e:
        return "Traceback:\n" + traceback.format_exc()

if __name__ == '__main__':
    with app.app_context():
        print(run_test())
