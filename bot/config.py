import os

             
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

                                                         
ML_API_URL = os.environ.get("ML_API_URL", "http://ml:8000")

          
ML_TIMEOUT_SEC = float(os.environ.get("ML_TIMEOUT_SEC", "30"))
