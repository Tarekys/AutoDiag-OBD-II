import xgboost as xgb
import joblib
import sqlite3
from utilize import fill_missing, encode_categorical_columns
from twilio.rest import Client
import os
from dotenv import load_dotenv
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Twilio configuration
load_dotenv()
'''
# to_phone_number = ****
# twilio_account_sid = ******
# twilio_auth_token = *******
# twilio_messaging_service_sid = *******
# phone = ********
'''

MODEL_PATH = "car_fault_classifier.json"
ENCODERS_PATH = "encoders.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"

PREDICTION_LABELS = {
    3: 'No Fault',
    2: 'Engine Fault',
    0: 'Electrical Fault',
    1: 'Emission Fault',
    4: 'Transmission Fault'
}

def get_prediction_message(prediction):
    messages = {
        0: "⚠️❗⚡ تحذير ❗: تم رصد احتمال حدوث خلل كهربائي قريبًا. يُوصى بالتحقق من الأنظمة الكهربائية.",
        1: "⚠️❗🌫️ انتباه❗: هناك مؤشرات على احتمالية وجود مشكلة في نظام الانبعاثات.",
        2: "⚠️❗🔧 تحذير❗: تم رصد احتمال وجود خلل في أجزاء من المحرك.",
        3: "✅ النظام يعمل بسلاسة حاليًا ولا توجد أعطال متوقعة.",
        4: "⚠️❗⚙️ انتباه عاجل❗: احتمال بحدوث خلل في ناقل الحركة خلال دقائق."
    }
    return messages.get(prediction, "❗ نوع العطل غير معروف، يُرجى المراجعة.")


def can_send_sms(fault_code, conn, cur):


    current_time = time.time()
    cur.execute("SELECT last_sent_time FROM sms_log WHERE fault_code = ?", (fault_code,))
    result = cur.fetchone()
    last_sent_time = result[0] if result else 0
    time_diff = current_time - last_sent_time
    logging.info(f"فحص العطل {fault_code}: الفارق الزمني = {time_diff:.2f} ثانية")
    return time_diff >= 120 


def update_sms_log(fault_code, conn, cur):

    current_time = time.time()
    cur.execute("""
        INSERT OR REPLACE INTO sms_log (fault_code, last_sent_time)
        VALUES (?, ?)
    """, (fault_code, current_time))
    conn.commit()
    logging.info(f"تم تحديث sms_log للعطل {fault_code} بوقت {current_time}")


# Function to preprocess data and make predictions
def preprocess_and_predict_from_df(original_data):
    try:
        original_data.columns = original_data.columns.str.strip()
        data = original_data.copy()
        logging.info(f"loading....{len(data)} row of data...")

        data = fill_missing(data, strategy_numeric='auto', save_indicators=False)
        encoded_data, _ = encode_categorical_columns(data, encoders_path=ENCODERS_PATH)

       # lodding feature columns
        expected_columns = joblib.load(FEATURE_COLUMNS_PATH)
        for col in expected_columns:
            if col not in encoded_data.columns:
                encoded_data[col] = 0
        prediction_data = encoded_data[expected_columns]

        # model loading and prediction
        logging.info("loading prediction....")
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)

        predictions = model.predict(prediction_data)
        logging.info(f"Prediction done at {len(predictions)}")

        # merge predictions with original data
        original_data['Predicted_Fault'] = [PREDICTION_LABELS.get(p, 'Unknown Fault') for p in predictions]
        original_data['Prediction_Message'] = [get_prediction_message(p) for p in predictions]

        conn = sqlite3.connect("obd_data.db")
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS sms_log (
                fault_code INTEGER PRIMARY KEY,
                last_sent_time REAL
            )
        """)
        conn.commit()


        logging.info(" جاري التحقق من الأعطال لإرسال رسائل SMS...")

        unique_faults = set(predictions) - {3}  # "No Fault"

        client = Client('''twilio_account_sid''', '''twilio_auth_token''')
        to_phone_number = '''phone'''

        for fault_code in unique_faults:
            fault_label = PREDICTION_LABELS.get(fault_code, 'Unknown Fault')
            if not can_send_sms(fault_code, conn, cur):
                logging.info(f" تم تجاهل إرسال رسالة للعطل '{fault_label}' لأنها أُرسلت خلال الدقيقة الماضية.")
                continue

            message_body = f"تنبيه: تم اكتشاف عطل من نوع {fault_label}. {get_prediction_message(fault_code)}"
            try:
                message = client.messages.create(
                    messaging_service_sid='''twilio_messaging_service_sid''',
                    body=message_body,
                    to=to_phone_number
                )
                logging.info(f"SMS message for the fault '{fault_label}' has been sent successfully: {message.sid}")
                update_sms_log(fault_code, conn, cur)
            except Exception as sms_error:
                logging.error(f"Error sending SMS message for the fault'{fault_label}': {str(sms_error)}")

#==========================================================
        # Insert data into SQLite database
        logging.info("saved.......")
        for _, row in original_data.iterrows():
            cur.execute("""
                INSERT INTO obd_data (
                    Engine_RPM, Coolant_Temp_C, Oil_Temp_C, Idle_Status,
                    Engine_Load_Percent, Ignition_Timing_Deg, MAP_kPa, MAF_gps, Battery_Voltage_V,
                    Charging_System_Status, O2_Sensor_V, Catalytic_Converter_Percent, EGR_Status, Vehicle_Speed_kmh,
                    Transmission_Gear, Brake_Status, Tire_Pressure_psi, Ambient_Temp_C, Battery_Age_Months,
                    Fuel_Level_Percent, Predicted_Fault, Prediction_Message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['Engine_RPM'], row['Coolant_Temp_C'], row['Oil_Temp_C'], row['Idle_Status'],
                row['Engine_Load_Percent'], row['Ignition_Timing_Deg'], row['MAP_kPa'], row['MAF_gps'], row['Battery_Voltage_V'],
                row['Charging_System_Status'], row['O2_Sensor_V'], row['Catalytic_Converter_Percent'], row['EGR_Status'], row['Vehicle_Speed_kmh'],
                row['Transmission_Gear'], row['Brake_Status'], row['Tire_Pressure_psi'], row['Ambient_Temp_C'], row['Battery_Age_Months'],
                row['Fuel_Level_Percent'], row['Predicted_Fault'], row['Prediction_Message']
            ))

        conn.commit()
        conn.close()
        logging.info("✅ Save to SQLite completed successfully.")

        return predictions, original_data

    except Exception as e:
        logging.error(f" Error in SQLite: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None