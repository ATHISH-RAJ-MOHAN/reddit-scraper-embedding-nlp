import requests
import time
import random
from config.headers import get_headers

REQUEST_COUNT = 0
MAX_REQUESTS = 500

# Create a session to persist cookies (Crucial for Reddit)
session = requests.Session()
session.headers.update(get_headers())

def fetch_html(url):
    global REQUEST_COUNT

    if REQUEST_COUNT >= MAX_REQUESTS:
        print("Reached 500 requests. Cooling down for 60 seconds...")
        time.sleep(60)
        REQUEST_COUNT = 0

    for attempt in range(5):
        try:
            # Use the session instead of requests.get
            response = session.get(url, timeout=10)

            if response.status_code == 200:
                REQUEST_COUNT += 1
                time.sleep(random.uniform(2, 4)) # Slightly increased delay to be safer
                return response.text

            elif response.status_code == 429:
                print("Rate limited. Sleeping 30 seconds...")
                time.sleep(30)
            
            else:
                print(f"Failed with status code: {response.status_code}")
                time.sleep(5)

        except requests.exceptions.Timeout:
            print("Timeout. Retrying...")
            time.sleep(5)

        except Exception as e:
            print("Error:", e)
            time.sleep(5)

    return None