import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
from datetime import datetime
import random
from streamlit_autorefresh import st_autorefresh
from charts_module import ChartGenerator

# API Configuration
FASTAPI_URL = "********************" 

# Streamlit configuration
st.set_page_config(
    page_title="Vehicle Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language settings
LANGUAGES = {
    'العربية': 'ar',
    'English': 'en'
}

TRANSLATIONS = {
    'ar': {
        'title': "تحليل بيانات المركبة والتنبؤ بالأعطال",
        'subtitle': 'منصة شاملة لتحليل بيانات المركبات في الوقت الفعلي والتنبؤ بالأعطال',
        'chart_selection': 'اختيار الرسوم البيانية',
        'chart_size': 'حجم الرسوم البيانية',
        'language_selection': 'اختيار اللغة',
        'sidebar_title': 'إعدادات التحكم',
        'select_charts': 'اختر الرسوم البيانية التي تريد عرضها',
        'no_charts_selected': 'لم يتم اختيار أي رسم بياني',
        'vehicle_dashboard': 'لوحة تحكم المركبة',
        'error_loading': 'خطأ في تحميل البيانات: ',
        'small': 'صغير',
        'medium': 'متوسط',
        'large': 'كبير',
        'fault_types': 'أنواع الأعطال المحتملة',
        'select_fault': 'اختر نوع العطل لعرضه:',
        'all_types': 'جميع الأنواع',
        'download_results': 'تحميل النتائج كملف CSV',
        'no_results': 'لا توجد نتائج لعرضها',
        'api_error': 'حدث خطأ من الخادم: ',
        'processing_error': 'حدث خطأ أثناء المعالجة: ',
        'processed_success': 'تمت المعالجة والتنبؤ بنجاح! ✅',
        'welcome': 'مرحبًا بك في تحليل بيانات المركبة',
        'simulator_status': 'حالة المحاكي: '
    },
    'en': {
        'title': 'Vehicle Analytics and Fault Prediction',
        'subtitle': 'A comprehensive platform for real-time vehicle data analysis and fault prediction',
        'chart_selection': 'Chart Selection',
        'chart_size': 'Chart Size',
        'language_selection': 'Language Selection',
        'sidebar_title': 'Control Settings',
        'select_charts': 'Select the charts you want to display',
        'no_charts_selected': 'No charts selected',
        'vehicle_dashboard': 'Vehicle Dashboard',
        'error_loading': 'Error loading data: ',
        'small': 'Small',
        'medium': 'Medium',
        'large': 'Large',
        'fault_types': 'Types of Possible Faults',
        'select_fault': 'Select the type of fault to view:',
        'all_types': 'All types',
        'download_results': 'Download results as CSV',
        'no_results': 'No results to display',
        'api_error': 'Server error occurred: ',
        'processing_error': 'Error occurred during processing: ',
        'processed_success': 'Processed and predicted successfully! ✅',
        'welcome': 'Welcome to Vehicle Analytics',
        'simulator_status': 'Simulator Status: '
    }
}

# CSS style
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@200;300;400;500;700;800;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Almarai:wght@300;400;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700;800&display=swap');
        
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .stApp {
            font-family: 'Cairo', 'Tajawal', 'Almarai', sans-serif;
            font-weight: 400;
            letter-spacing: 0.5px;
            line-height: 1.6;
        }
        
        .hero-header {
            background: linear-gradient(rgba(30, 60, 114, 0.7), rgba(42, 82, 152, 0.7)),
                       url('https://images.unsplash.com/photo-1542362567-b07e54358753?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
            background-size: cover;
            background-position: center;
            padding: 3rem 2rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .hero-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            opacity: 0.1;
        }
        
        .hero-title {
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
            font-size: 3.5rem;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
            letter-spacing: 1px;
        }
        
        .hero-subtitle {
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
            font-size: 1.3rem;
            opacity: 0.9;
            position: relative;
            z-index: 1;
            letter-spacing: 0.5px;
        }
        
        .css-1d391kg {
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        }
        
        .sidebar-content {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
        }
        
        .metric-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin: 1rem 0;
            transition: transform 0.3s ease;
            border-left: 5px solid #3498db;
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        
        .chart-container {
            background: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            margin: 2rem 0;
            border: 1px solid rgba(0,0,0,0.05);
            font-family: 'Cairo', sans-serif;
        }
        
        .chart-title {
            font-family: 'Cairo', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #3498db !important;  /* تم تغيير اللون من blue إلى #3498db الأزرق الأوضح */
            margin-bottom: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #3498db;  /* تم تغيير لون الخط السفلي أيضاً إلى الأزرق */
            letter-spacing: 0.5px;
        }
        
        .chart-description {
            font-family: 'Tajawal', sans-serif;
            font-size: 1rem;
            color: black;
            margin-bottom: 1rem;
            font-weight: 400;
            line-height: 1.7;
        }
        
        .upload-area {
            background: white;
            padding: 3rem 2rem;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            margin: 2rem 0;
            text-align: center;
            border: 2px dashed #3498db;
            transition: all 0.3s ease;
            font-family: 'Cairo', sans-serif;
        }
        
        .upload-area:hover {
            border-color: #e74c3c;
            transform: translateY(-2px);
        }
        
        .status-success {
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            color: white;
            padding: 1rem 2rem;
            border-radius: 10px;
            margin: 1rem 0;
            font-family: 'Cairo', sans-serif;
            font-weight: 500;
        }
        
        .status-warning {
            background: linear-gradient(135deg, #f39c12, #e67e22);
            color: white;
            padding: 1rem 2rem;
            border-radius: 10px;
            margin: 1rem 0;
            font-family: 'Cairo', sans-serif;
            font-weight: 500;
        }
        
        .fault-table {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin: 2rem 0;
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
        }

        .fault-table table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }

        .fault-table th {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 12px;
            font-weight: 800;
            text-align: center;
        }

        .fault-table td {
            padding: 12px;
            border-bottom: 1px solid #e1e8ed;
            text-align: center;
            font-weight: 800;
        }

        .fault-table tr:last-child td {
            border-bottom: none;
        }

        .fault-table tr:nth-child(even) {
            background: #f8f9fa;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .footer {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-top: 3rem;
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in-up {
            animation: fadeInUp 0.6s ease-out;
        }
        
        .rtl {
            direction: rtl;
            text-align: right;
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 800;
            transition: all 0.3s ease;
            font-family: 'Cairo', sans-serif;
            letter-spacing: 0.5px;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .stSelectbox > div > div {
            border-radius: 10px;
            border: 2px solid #e1e8ed;
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
        }
        
        .stMultiSelect > div > div {
            border-radius: 10px;
            border: 2px solid #e1e8ed;
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Cairo', sans-serif !important;
            font-weight: 800;
            letter-spacing: 0.5px;
            color: #3498db !important;  
        }
        
        h1 {
            font-weight: 800;
        }
        
        p {
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
            line-height: 1.7;
            letter-spacing: 0.3px;
        }
        
        label {
            font-family: 'Cairo', sans-serif !important;
            font-weight: 800;
        }
        
        .stDataFrame {
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
        }
        
        .metric-container .metric-label {
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
        }
        
        .metric-container .metric-value {
            font-family: 'Cairo', sans-serif;
            font-weight: 800;
        }
        
        /* تم إضافة هذا التعديل لضمان تطبيق اللون الأزرق على العناوين في الشريط الجانبي */
        .sidebar-content h2, .sidebar-content h3 {
            color: #3498db !important;
        }
        
        .metric-card h3 {
            color: #3498db !important;
        }
    </style>
    """, unsafe_allow_html=True)

def generate_row():

    is_fault = random.random() > 0.85  # n% Fault
    status = "Fault" if is_fault else "Normal"

    def choose(normal_range, fault_range, is_float=False, round_to=0):
        if is_fault:
            val = np.random.uniform(*fault_range) if is_float else np.random.randint(*fault_range)
        else:
            val = np.random.uniform(*normal_range) if is_float else np.random.randint(*normal_range)
        return round(val, round_to) if is_float else val

    return {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Engine_RPM": choose((900, 2000), (4000, 6000)),
        "Coolant_Temp_C": choose((85, 95), (100, 120)),
        "Oil_Temp_C": choose((80, 95), (110, 130)),
        "Idle_Status": np.random.choice(["False", "True"], p=[0.8, 0.2]),
        "Engine_Load_Percent": choose((25, 50), (80, 100)),
        "Ignition_Timing_Deg": choose((5, 20), (-5, 0)),
        "MAP_kPa": choose((30, 60), (80, 100)),
        "MAF_gps": choose((5, 15), (60, 150), is_float=True, round_to=1),
        "Battery_Voltage_V": choose((13.5, 14.2), (11.0, 12.0), is_float=True, round_to=1),
        "Charging_System_Status": np.random.choice(["Normal", "Fault"], p=[0.9, 0.1]),
        "O2_Sensor_V": choose((0.6, 0.8), (0.1, 0.2), is_float=True, round_to=2),
        "Catalytic_Converter_Percent": choose((90, 99), (70, 80)),
        "EGR_Status": np.random.choice(["Open", "Closed", "Stuck_Open"], p=[0.5, 0.4, 0.1]),
        "Vehicle_Speed_kmh": choose((40, 90), (150, 200)),
        "Transmission_Gear": np.random.choice(
            ["P", "R", "N", "D", "1", "2", "3", "4", "5", "6"],
            p=[0.1, 0.05, 0.05, 0.5, 0.05, 0.05, 0.07, 0.07, 0.04, 0.02]
        ),
        "Brake_Status": np.random.choice(["Released", "Engaged"], p=[0.85, 0.15]),
        "Tire_Pressure_psi": choose((30, 34), (20, 26)),
        "Ambient_Temp_C": choose((20, 30), (35, 40)),
        "Battery_Age_Months": choose((6, 24), (48, 72), is_float=True, round_to=1),
        "Fuel_Level_Percent": choose((50, 100), (0, 15)),
        "Status": status
    }

'''
Function to simulate data generation like OBD-II ELM327 
OBD-II ELM327 device - It was handed over to the university.
Instead, I simulated it running so you could see it.

'''
def simulate_data():
    if st.session_state.get('simulator_on', False):
        with st.spinner("Generating simulated data..."):
            new_data = pd.DataFrame([generate_row()])
            if not new_data.empty:
                st.session_state['simulated_data'] = pd.concat([st.session_state['simulated_data'], new_data], ignore_index=True)
                
                # Send to API
                files = {'file': ('simulated_data.csv', st.session_state['simulated_data'].tail(1).to_csv(index=False), 'text/csv')}
                response = requests.post(FASTAPI_URL, files=files)
                if response.status_code == 200 and response.json().get("status") == "success":
                    new_prediction = pd.DataFrame(response.json()["results"])
                    st.session_state['predictions'] = pd.concat([st.session_state['predictions'], new_prediction], ignore_index=True)
                else:
                    t = TRANSLATIONS[st.session_state['lang_code']]
                    st.error(f"{t['api_error']} {response.status_code}")

def main():
    load_css()
    
    # Initialize session state
    if 'lang_code' not in st.session_state:
        st.session_state['lang_code'] = 'ar'
    if 'simulated_data' not in st.session_state:
        st.session_state['simulated_data'] = pd.DataFrame(columns=[
            'Timestamp', 'Engine_RPM', 'Coolant_Temp_C', 'Oil_Temp_C', 'Idle_Status',
            'Engine_Load_Percent', 'Ignition_Timing_Deg', 'MAP_kPa', 'MAF_gps',
            'Battery_Voltage_V', 'Charging_System_Status', 'O2_Sensor_V',
            'Catalytic_Converter_Percent', 'EGR_Status', 'Vehicle_Speed_kmh',
            'Transmission_Gear', 'Brake_Status', 'Tire_Pressure_psi', 'Ambient_Temp_C',
            'Battery_Age_Months', 'Fuel_Level_Percent', 'Status'
        ])
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = pd.DataFrame(columns=['Recording', 'Predicted_Fault', 'Prediction_Message'])

    # Auto-refresh when simulator is on
    if st.session_state.get('simulator_on', False):
        st_autorefresh(interval=5000, key="data_refresh")  # Refresh every 5 seconds

    # Sidebar settings
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        selected_language = st.selectbox(
            "Language 🌐",
            options=list(LANGUAGES.keys()),

            index=0 if st.session_state['lang_code'] == 'ar' else 1,

            key="language_select"
        )
        
        st.session_state['lang_code'] = LANGUAGES[selected_language]
        t = TRANSLATIONS[st.session_state['lang_code']]
        
        st.markdown(f"<h2 style='text-align: center; color: #3498db;'>{t['sidebar_title']}</h2>", unsafe_allow_html=True)  # تم تغيير اللون هنا إلى الأزرق
        
        if 'simulator_on' not in st.session_state:
            st.session_state['simulator_on'] = False
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("TURN ON OBD-II ELM327🟢", key="obd_on"):
                st.session_state['simulator_on'] = True
        with col2:
            if st.button("TURN OFF OBD-II ELM327🟠", key="obd_off"):
                st.session_state['simulator_on'] = False
        
        st.markdown(f"<p>{t['simulator_status']} {'ON' if st.session_state['simulator_on'] else 'OFF'}</p>", unsafe_allow_html=True)
        

        # Chart settings
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.markdown(f"<h3 style='color: #3498db;'>📈{t['chart_size']}</h3>", unsafe_allow_html=True)  
        chart_size = st.selectbox(
            "Size",
            options=[t['small'], t['medium'], t['large']],
            index=1,
            key="chart_size_select"
        )
        
        size_mapping = {
            t['small']: 250,
            t['medium']: 400,
            t['large']: 500
        }
        chart_height = size_mapping[chart_size]
        
        st.markdown(f"<h3 style='color: #3498db;'>📈{t['chart_selection']}</h3>", unsafe_allow_html=True)  
        chart_generator = ChartGenerator(st.session_state['lang_code'])
        available_charts = chart_generator.get_available_charts()
        
        selected_charts = st.multiselect(
            t['select_charts'],
            options=list(available_charts.keys()),
            default=[],
            key="chart_select",
            help="Select multiple charts to display"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main header
    st.markdown(f'''
    <div class="hero-header fade-in-up {'rtl' if st.session_state['lang_code'] == 'ar' else ''}">
        <h1 class="hero-title">{t['title']}</h1>
        <p class="hero-subtitle">{t['subtitle']}</p>
    </div>
    ''', unsafe_allow_html=True)
    

    # Run simulation if enabled
    if st.session_state.get('simulator_on', False):
        simulate_data()

    # Display prediction table for last 10 predictions
    if not st.session_state['predictions'].empty:
        st.markdown(f"<h2 style='color: #3498db;'>{t['fault_types']}</h2>", unsafe_allow_html=True) 
        fault_types = [t['all_types']] + list(st.session_state['predictions']['Predicted_Fault'].unique())
        selected_fault = st.selectbox(t['select_fault'], fault_types, key="fault_select")
        
        if selected_fault != t['all_types']:
            filtered_df = st.session_state['predictions'][st.session_state['predictions']['Predicted_Fault'] == selected_fault].tail(10)
        else:
            filtered_df = st.session_state['predictions'].tail(10)
        
        if not filtered_df.empty:
            table_data = []
            for idx, row in filtered_df.iterrows():
                table_data.append({
                    'Recording': f"{idx + 1}",
                    'Predicted_Fault': row['Predicted_Fault'], 
                    'Prediction_Message': row['Prediction_Message']  
                })
            table_df = pd.DataFrame(table_data)
            st.markdown('<div class="fault-table">', unsafe_allow_html=True)
            st.table(table_df)
            st.markdown('</div>', unsafe_allow_html=True)
            
            csv = table_df.to_csv(index=False)
            st.download_button(
                label=t['download_results'],
                data=csv,
                file_name="fault_predictions.csv",
                mime="text/csv",
                key="download_button"
            )
        else:
            st.markdown(f'<div class="status-warning">{t["no_results"]}</div>', unsafe_allow_html=True)

    # Display charts dynamically
    if not st.session_state['simulated_data'].empty and 'selected_charts' in locals() and selected_charts:
        st.markdown(f'''
        <div class="chart-container fade-in-up {'rtl' if st.session_state['lang_code'] == 'ar' else ''}">
            <h2 class="chart-title">{t['vehicle_dashboard']}</h2>
        </div>
        ''', unsafe_allow_html=True)
        
        charts_per_row = 2
        chart_rows = [selected_charts[i:i + charts_per_row] for i in range(0, len(selected_charts), charts_per_row)]
        
        for row in chart_rows:
            cols = st.columns(len(row))
            for i, chart_name in enumerate(row):
                with cols[i]:
                    placeholder_chart = st.empty()
                    with placeholder_chart.container():
                        result = chart_generator.create_chart(chart_name, st.session_state['simulated_data'], chart_height)
                        fig = result["fig"]
                        description = result["description"]
                        
                        st.markdown(f'''
                        <div class="chart-container fade-in-up {'rtl' if st.session_state['lang_code'] == 'ar' else ''}">
                            <h3 class="chart-title">{chart_name}</h3>
                            <p class="chart-description">{description}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Could not create chart: {chart_name}. Data missing or invalid.")

    # Display quick statistics and charts if no charts selected
    elif not st.session_state['simulated_data'].empty:
        st.markdown(f'''
        <div class="metric-card fade-in-up {'rtl' if st.session_state['lang_code'] == 'ar' else ''}">
            <h3 style="text-align: center; color: #7f8c8d;">
                {t['no_charts_selected']}
            </h3>
            <p style="text-align: center;">{t['select_charts']}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Welcome message if no data
    else:
        st.markdown(f'''
        <div class="upload-area fade-in-up {'rtl' if st.session_state['lang_code'] == 'ar' else ''}">
            <h2 style="color: #2c3e50;">{t.get('welcome', 'Welcome to Vehicle Analytics')}</h2>
            <p style="color: #7f8c8d; font-size: 1.1rem;">
                {t.get('welcome', 'Welcome to Vehicle Analytics')}
            </p>
            <br>
        </div>
        ''', unsafe_allow_html=True)
    
    # Footer
    st.markdown(f'''
    <div class="footer fade-in-up">
        <p style="opacity: 0.8;">Real-time Vehicle Data Analysis & Fault Prediction Platform</p>
        <p>2025 | Teem OHI | Powered by AI Engineering</p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()