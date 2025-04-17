import pandas as pd
import xgboost as xgb
import joblib
import sqlite3
import os
from utilize import fill_missing, encode_categorical_columns

MODEL_PATH = "car_fault_classifier.json"
ENCODERS_PATH = "encoders.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
DB_PATH = "OBD_Predictions.db"
TABLE_NAME = "fault_predictions"


PREDICTION_LABELS = {
    3: 'No Fault',
    2: 'Engine Fault',
    0: 'Electrical Fault',
    1: 'Emission Fault',
    4: 'Transmission Fault'
}

def get_prediction_message(prediction):
    messages = {
    0: "⚠️❗⚡ تحذير ❗: تم رصد احتمال حدوث خلل كهربائي قريبًا. يُوصى بالتحقق من الأنظمة الكهربائية. يمكنك استشارة المساعد الذكي عن الحلول الممكنة.",
    1: "⚠️❗🌫️ انتباه❗: هناك مؤشرات على احتمالية وجود مشكلة في نظام الانبعاثات. يمكنك استشارة المساعد الذكي عن الحلول الممكنة.",
    2: "⚠️❗🔧 تحذير❗: تم رصد احتمال وجود خلل في أجزاء من المحرك. يُفضل إجراء فحص فوري. يمكنك استشارة المساعد الذكي عن الحلول الممكنة.",
    3: "✅ النظام يعمل بسلاسة حاليًا ولا توجد أعطال متوقعة. يمكنك استشارة .",
    4: "⚠️❗⚙️ انتباه عاجل❗: احتمال بحدوث خلل في ناقل الحركة خلال دقائق. يمكنك استشارة المساعد الذكي عن الحلول الممكنة."
}

    return messages.get(prediction, "❗ نوع العطل غير معروف، يُرجى المراجعة.")

# Prediction and processing function
def preprocess_and_predict_from_df(original_data):
    """
    تستقبل DataFrame من Streamlit وتعيد النتائج بعد المعالجة والتنبؤ.
    """
    try:
        data = original_data.copy()
        print(f"جاري معالجة {len(data)} صف من البيانات...")

        # خطوات المعالجة
        data = fill_missing(data, strategy_numeric='auto', save_indicators=False)
        encoded_data, _ = encode_categorical_columns(data, encoders_path=ENCODERS_PATH)

        # تحميل الأعمدة المطلوبة
        expected_columns = joblib.load(FEATURE_COLUMNS_PATH)
        for col in expected_columns:
            if col not in encoded_data.columns:
                encoded_data[col] = 0
        prediction_data = encoded_data[expected_columns]

        # تحميل النموذج
        print("جاري تحميل النموذج وإجراء التنبؤ...")
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)

        predictions = model.predict(prediction_data)
        print(f"تم الانتهاء من التنبؤ. عدد التنبؤات: {len(predictions)}")

        # إضافة النتائج إلى البيانات الأصلية
        original_data['Predicted_Fault'] = [PREDICTION_LABELS.get(p, 'Unknown Fault') for p in predictions]
        original_data['Prediction_Message'] = [get_prediction_message(p) for p in predictions]

        # Save the predictions to database
        print("جاري حفظ النتائج في قاعدة البيانات...")
        save_to_database(original_data)
        print("تم حفظ النتائج بنجاح.")
        
        # إحصائيات سريعة
        fault_counts = {}
        for p in predictions:
            fault_name = PREDICTION_LABELS.get(p, 'Unknown')
            fault_counts[fault_name] = fault_counts.get(fault_name, 0) + 1
        
        print("\nملخص نتائج التنبؤ:")
        for fault, count in fault_counts.items():
            print(f"- {fault}: {count} ({count/len(predictions)*100:.1f}%)")
        
        return predictions, original_data

    except Exception as e:
        print(f"حدث خطأ أثناء التنبؤ: {str(e)}")
        import traceback
        traceback.print_exc()  
        return None, None

# SQLite
def save_to_database(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
    conn.close()