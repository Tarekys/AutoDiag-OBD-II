import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib

def fill_missing(data, strategy_numeric='auto', save_indicators=False):
    """
    ملء القيم المفقودة في البيانات مع إضافة أعمدة تشير إلى القيم المفقودة (اختياري).

    """
    for col in data.columns:
        if data[col].isnull().sum() == 0:
            continue

        try:
            # إضافة عمود مؤشر للقيم المفقودة (إذا طُلب)
            if save_indicators:
                data[f'is_missing_{col}'] = data[col].isnull().astype(int)

            # معالجة الأعمدة الرقمية
            if data[col].dtype in ['float64', 'int64', 'bool']:
                if strategy_numeric == 'auto':
                    # اختيار median إذا كان التوزيع منحرفًا
                    skew_value = data[col].skew()
                    if pd.notna(skew_value) and skew_value > 1:
                        data[col] = data[col].fillna(data[col].median())
                    else:
                        data[col] = data[col].fillna(data[col].mean())
                elif strategy_numeric == 'median':
                    data[col] = data[col].fillna(data[col].median())
                elif strategy_numeric == 'mean':
                    data[col] = data[col].fillna(data[col].mean())
                else:
                    raise ValueError("strategy_numeric must be 'mean', 'median', or 'auto'")
            
            # معالجة الأعمدة الفئوية
            else:
                mode_val = data[col].mode()
                if not mode_val.empty:
                    data[col] = data[col].fillna(mode_val[0])
                else:
                    data[col] = data[col].fillna('Unknown')  # قيمة افتراضية إذا لم يكن هناك mode

        except Exception as e:
            print(f"خطأ في معالجة العمود {col}: {str(e)}")
            continue

    return data

def encode_categorical_columns(data, encoders_path=None):
    """
    ترميز الأعمدة الفئوية باستخدام LabelEncoder للأعمدة الثنائية وOneHotEncoder للأعمدة متعددة القيم.

    """
    encoded_data = data.copy()
    label_encoders = {}

    try:
        if encoders_path:
            # تحميل المحولات المحفوظة
            label_encoders = joblib.load(encoders_path)
            binary_cols = [col for col in label_encoders.keys() if col != 'onehot_encoder' and col != 'onehot_columns']
            multi_cols = label_encoders.get('onehot_columns', [])
            onehot_encoder = label_encoders.get('onehot_encoder', None)
        else:
            # تحديد الأعمدة الفئوية
            categorical_cols = encoded_data.select_dtypes(include=['object']).columns
            binary_cols = []
            multi_cols = []

            # تصنيف الأعمدة 
            for col in categorical_cols:
                unique_vals = encoded_data[col].nunique()
                if unique_vals == 2:
                    binary_cols.append(col)
                elif unique_vals > 2:
                    multi_cols.append(col)

        # LabelEncoder
        for col in binary_cols:
            if encoders_path and col in label_encoders:
                le = label_encoders[col]
                try:
                    encoded_data[col] = le.transform(encoded_data[col])
                except ValueError:
                    print(f"تحذير: القيم في العمود {col} تحتوي على قيم جديدة غير موجودة في التدريب")
                    encoded_data[col] = encoded_data[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
                    encoded_data[col] = le.transform(encoded_data[col])
            else:
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(encoded_data[col])
                label_encoders[col] = le

        # OneHotEncoder
        if multi_cols:
            if encoders_path and onehot_encoder:
                try:
                    onehot_encoded = onehot_encoder.transform(encoded_data[multi_cols])
                    onehot_cols = onehot_encoder.get_feature_names_out(multi_cols)
                except ValueError:
                    print(f"تحذير: القيم في الأعمدة {multi_cols} تحتوي على قيم جديدة")
                    for col in multi_cols:
                        encoded_data[col] = encoded_data[col].map(lambda x: x if x in onehot_encoder.categories_[multi_cols.index(col)] else onehot_encoder.categories_[multi_cols.index(col)][0])
                    onehot_encoded = onehot_encoder.transform(encoded_data[multi_cols])
                    onehot_cols = onehot_encoder.get_feature_names_out(multi_cols)
            else:
                onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
                onehot_encoded = onehot_encoder.fit_transform(encoded_data[multi_cols])
                onehot_cols = onehot_encoder.get_feature_names_out(multi_cols)
                label_encoders['onehot_encoder'] = onehot_encoder
                label_encoders['onehot_columns'] = multi_cols

            # إنشاء DataFrame للبيانات المحولة
            onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_cols, index=encoded_data.index)

            # إزالة الأعمدة الأصلية ودمج البيانات المحولة
            encoded_data = encoded_data.drop(columns=multi_cols)
            encoded_data = pd.concat([encoded_data, onehot_df], axis=1)

        # حفظ المحولات فقط في وضع التدريب (عندما لا يتم تمرير encoders_path)
        if not encoders_path:
            joblib.dump(label_encoders, 'encoders.pkl')
            print("تم حفظ المحولات باسم 'encoders.pkl'")

        return encoded_data, label_encoders

    except Exception as e:
        print(f"Categorical column encoding error: {str(e)}")
        return encoded_data, label_encoders