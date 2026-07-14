import logging
import sqlite3
import time

import joblib
import xgboost as xgb
from twilio.rest import Client

from car_fault_prediction.config.settings import (
    ENCODERS_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    RUNTIME_DATABASE_PATH,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_MESSAGING_SERVICE_SID,
    TWILIO_PHONE_NUMBER,
)
from car_fault_prediction.utils.preprocessing import (
    encode_categorical_columns,
    fill_missing,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PREDICTION_LABELS = {
    3: "No Fault",
    2: "Engine Fault",
    0: "Electrical Fault",
    1: "Emission Fault",
    4: "Transmission Fault",
}


def get_prediction_message(prediction):
    messages = {
        0: "⚠️❗⚡ تحذير ❗: تم رصد احتمال حدوث خلل كهربائي قريبًا. يُوصى بالتحقق من الأنظمة الكهربائية.",
        1: "⚠️❗🌫️ انتباه❗: هناك مؤشرات على احتمالية وجود مشكلة في نظام الانبعاثات.",
        2: "⚠️❗🔧 تحذير❗: تم رصد احتمال وجود خلل في أجزاء من المحرك.",
        3: "✅ النظام يعمل بسلاسة حاليًا ولا توجد أعطال متوقعة.",
        4: "⚠️❗⚙️ انتباه عاجل❗: احتمال بحدوث خلل في ناقل الحركة خلال دقائق.",
    }
    return messages.get(prediction, "❗ نوع العطل غير معروف، يُرجى المراجعة.")


def can_send_sms(fault_code, connection, cursor):
    current_time = time.time()
    cursor.execute("SELECT last_sent_time FROM sms_log WHERE fault_code = ?", (fault_code,))
    result = cursor.fetchone()
    last_sent_time = result[0] if result else 0
    time_diff = current_time - last_sent_time
    logger.info("فحص العطل %s: الفارق الزمني = %.2f ثانية", fault_code, time_diff)
    return time_diff >= 120


def update_sms_log(fault_code, connection, cursor):
    current_time = time.time()
    cursor.execute(
        """
        INSERT OR REPLACE INTO sms_log (fault_code, last_sent_time)
        VALUES (?, ?)
        """,
        (fault_code, current_time),
    )
    connection.commit()
    logger.info("تم تحديث sms_log للعطل %s بوقت %s", fault_code, current_time)


def preprocess_and_predict_from_df(original_data):
    try:
        original_data.columns = original_data.columns.str.strip()
        data = original_data.copy()
        logger.info("loading....%s row of data...", len(data))

        data = fill_missing(data, strategy_numeric="auto", save_indicators=False)
        encoded_data, _ = encode_categorical_columns(data, encoders_path=ENCODERS_PATH)

        expected_columns = joblib.load(FEATURE_COLUMNS_PATH)
        for column_name in expected_columns:
            if column_name not in encoded_data.columns:
                encoded_data[column_name] = 0
        prediction_data = encoded_data[expected_columns]

        logger.info("loading prediction....")
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)

        predictions = model.predict(prediction_data)
        logger.info("Prediction done at %s", len(predictions))

        original_data["Predicted_Fault"] = [
            PREDICTION_LABELS.get(prediction, "Unknown Fault") for prediction in predictions
        ]
        original_data["Prediction_Message"] = [
            get_prediction_message(prediction) for prediction in predictions
        ]

        connection = sqlite3.connect(RUNTIME_DATABASE_PATH)
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sms_log (
                fault_code INTEGER PRIMARY KEY,
                last_sent_time REAL
            )
            """
        )
        connection.commit()

        logger.info(" جاري التحقق من الأعطال لإرسال رسائل SMS...")

        unique_faults = set(predictions) - {3}

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        for fault_code in unique_faults:
            fault_label = PREDICTION_LABELS.get(fault_code, "Unknown Fault")
            if not can_send_sms(fault_code, connection, cursor):
                logger.info(
                    " تم تجاهل إرسال رسالة للعطل '%s' لأنها أُرسلت خلال الدقيقة الماضية.",
                    fault_label,
                )
                continue

            message_body = f"تنبيه: تم اكتشاف عطل من نوع {fault_label}. {get_prediction_message(fault_code)}"
            try:
                message = client.messages.create(
                    messaging_service_sid=TWILIO_MESSAGING_SERVICE_SID,
                    body=message_body,
                    to=TWILIO_PHONE_NUMBER,
                )
                logger.info(
                    "SMS message for the fault '%s' has been sent successfully: %s",
                    fault_label,
                    message.sid,
                )
                update_sms_log(fault_code, connection, cursor)
            except Exception as sms_error:
                logger.error(
                    "Error sending SMS message for the fault '%s': %s",
                    fault_label,
                    sms_error,
                )

        logger.info("saved.......")
        for _, row in original_data.iterrows():
            cursor.execute(
                """
                INSERT INTO obd_data (
                    Engine_RPM, Coolant_Temp_C, Oil_Temp_C, Idle_Status,
                    Engine_Load_Percent, Ignition_Timing_Deg, MAP_kPa, MAF_gps, Battery_Voltage_V,
                    Charging_System_Status, O2_Sensor_V, Catalytic_Converter_Percent, EGR_Status, Vehicle_Speed_kmh,
                    Transmission_Gear, Brake_Status, Tire_Pressure_psi, Ambient_Temp_C, Battery_Age_Months,
                    Fuel_Level_Percent, Predicted_Fault, Prediction_Message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["Engine_RPM"],
                    row["Coolant_Temp_C"],
                    row["Oil_Temp_C"],
                    row["Idle_Status"],
                    row["Engine_Load_Percent"],
                    row["Ignition_Timing_Deg"],
                    row["MAP_kPa"],
                    row["MAF_gps"],
                    row["Battery_Voltage_V"],
                    row["Charging_System_Status"],
                    row["O2_Sensor_V"],
                    row["Catalytic_Converter_Percent"],
                    row["EGR_Status"],
                    row["Vehicle_Speed_kmh"],
                    row["Transmission_Gear"],
                    row["Brake_Status"],
                    row["Tire_Pressure_psi"],
                    row["Ambient_Temp_C"],
                    row["Battery_Age_Months"],
                    row["Fuel_Level_Percent"],
                    row["Predicted_Fault"],
                    row["Prediction_Message"],
                ),
            )

        connection.commit()
        connection.close()
        logger.info("✅ Save to SQLite completed successfully.")

        return predictions, original_data
    except Exception as error:
        logger.exception(" Error in SQLite: %s", error)
        return None, None
