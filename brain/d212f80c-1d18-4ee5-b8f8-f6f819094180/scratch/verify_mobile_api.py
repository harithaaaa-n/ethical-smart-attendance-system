import requests
import json

BASE_URL = "http://127.0.0.1:5000/api/mobile"

def test_mobile_api():
    print("Testing Mobile API...")
    
    # 1. Test Ping
    try:
        r = requests.get(f"{BASE_URL}/ping")
        print(f"Ping: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"Ping Failed: {e}")

    # 2. Test Login (Student)
    # Using one of the students from the dummy data (look at database_manager.py)
    # Student username: name.replace(" ", "").lower(), password: student123
    # Let's try 'admin' first since we know it exists.
    login_data = {
        'username': 'admin',
        'password': 'admin123'
    }
    
    try:
        r = requests.post(f"{BASE_URL}/login", json=login_data)
        print(f"Login (Admin): {r.status_code}")
        if r.status_code == 200:
            user = r.json()
            print(f"User: {user['full_name']} ({user['role']})")
            
            # 3. Test Profile
            user_id = user['id']
            r_prof = requests.get(f"{BASE_URL}/profile?user_id={user_id}")
            print(f"Profile: {r_prof.status_code}")
            
            # 4. Test Timetable
            r_tt = requests.get(f"{BASE_URL}/timetable?user_id={user_id}")
            print(f"Timetable: {r_tt.status_code} - Found {len(r_tt.json())} entries")
            
    except Exception as e:
        print(f"Login/API Test Failed: {e}")

if __name__ == "__main__":
    test_mobile_api()
