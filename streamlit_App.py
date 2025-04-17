import streamlit as st
import pandas as pd
import requests
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# تحديد عنوان الـ API (سيتم ضبطه لاحقًا على Railway)
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict/")

# Streamlit configuration
st.set_page_config(
    page_title="Vehicle Dashboard",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS (نفس التنسيقات بدون تغيير)
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1e3a8a;
        font-family: 'Cairo', 'Arial', sans-serif;
        font-size: 2.5rem !important;
    }
    h2 {
        color: #1e3a8a;
        font-family: 'Cairo', 'Arial', sans-serif;
        font-size: 2rem !important;
    }
    h3 {
        color: #1e3a8a;
        font-family: 'Cairo', 'Arial', sans-serif;
        font-size: 1.5rem !important;
    }
    h4 {
        color: #1e3a8a;
        font-family: 'Cairo', 'Arial', sans-serif;
        font-size: 1.3rem !important;
    }
    .header-container {
        background-color: #1e3a8a;
        background-image: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        font-size: 1.2rem !important;
    }
    .upload-container, .chart-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border-right: 4px solid #3b82f6;
        font-size: 1.1rem !important;
    }
    .chart-option {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        margin: 0.5rem;
        text-align: center;
        transition: all 0.3s;
    }
    .chart-option:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .chart-option h4 {
        color: #1e3a8a;
        margin-bottom: 0.5rem;
        font-size: 1.3rem !important;
    }
    .chart-option p {
        color: #334155;
        font-size: 1rem !important;
    }
    .selected {
        border: 2px solid #3b82f6;
        background-color: #dbeafe;
    }
    .chart-options-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 1rem;
    }
    .expander-content {
        padding: 15px;
        background-color: #f9fafb;
        border-radius: 5px;
        font-size: 1.1rem !important;
    }
    .stExpander {
        background-color: white;
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .stSelectbox, .stSelectbox > div {
        margin-bottom: 20px;
        color: #1e3a8a !important;
        font-size: 1.2rem !important;
    }
    .stSelectbox > div[data-baseweb="select"] > div {
        background-color: white !important;
        border-color: #d1d5db !important;
        color: #1e293b !important;
        font-size: 1.2rem !important;
    }
    .stSelectbox div[role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }
    .stSelectbox div[role="option"] {
        color: #1e293b !important;
        background-color: #ffffff !important;
        font-size: 1.2rem !important;
        padding: 10px !important;
    }
    .stSelectbox div[role="option"]:hover {
        background-color: #dbeafe !important;
        color: #1e3a8a !important;
    }
    .dataframe {
        font-family: 'Cairo', 'Arial', sans-serif;
        width: 100%;
        border-collapse: collapse;
    }
    .dataframe th {
        background-color: #1e3a8a;
        color: white;
        padding: 12px 15px;
        text-align: right;
        font-size: 1.2rem !important;
    }
    .dataframe td {
        padding: 10px 15px;
        border-bottom: 1px solid #e5e7eb;
        text-align: right;
        color: #334155 !important;
        font-size: 1.1rem !important;
    }
    .dataframe tr:nth-child(even) {
        background-color: #f9fafb;
    }
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #1e3a8a;
        background-color: #f8fafc;
        border-radius: 5px;
        padding: 8px 12px !important;
        font-size: 1.2rem !important;
    }
    .streamlit-expanderContent {
        background-color: #ffffff;
        color: #1e293b !important;
        padding: 15px;
        border-radius: 0 0 5px 5px;
        font-size: 1.1rem !important;
    }
    .fault-details {
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        background-color: #ffffff;
        border-right: 4px solid #3b82f6;
        color: #1e293b !important;
        font-size: 1.1rem !important;
    }
    .no-fault {
        border-right-color: #10b981;
    }
    .engine-fault {
        border-right-color: #f59e0b;
    }
    .electrical-fault {
        border-right-color: #3b82f6;
    }
    .emission-fault {
        border-right-color: #8b5cf6;
    }
    .transmission-fault {
        border-right-color: #ef4444;
    }
    .fault-message {
        padding: 10px;
        background-color: #f0f9ff;
        border-radius: 5px;
        margin-top: 10px;
        color: #0c4a6e !important;
        font-size: 1.1rem !important;
    }
    p, span, div, li {
        color: #1e293b !important;
        font-size: 1.1rem !important;
    }
    .stAlert {
        border-radius: 5px;
        font-size: 1.1rem !important;
    }
    .stChart {
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-container"><h1>Vehicle Data Plate ⚙</h1></div>', unsafe_allow_html=True)

# Description of charts
chart_descriptions = {
    "1. Histogram of Engine RPM": "رسم بياني يوضح توزيع دورات المحرك، مع تمييز الدورات العالية والمنخفضة",
    "2. Line Graph of Engine RPM over time": "مخطط زمني يظهر تغيرات دورات المحرك مع الوقت، مع تمييز ملون للدورات العالية",
    "3. Line Graph of Coolant Temperature": "مخطط درجة حرارة سائل التبريد على مدار الزمن، مع خط تحذير عند 105 درجة مئوية",
    "4. Histogram of Oil Temperature": "رسم بياني يوضح توزيع درجات حرارة الزيت مع إظهار المتوسط والوسيط",
    "5. Line Graph of Oil Temperature": "مخطط زمني يظهر تغيرات درجة حرارة الزيت على مدار اليوم",
    "6. Line Graph of Engine RPM and Oil Temperature": "مخطط مزدوج يظهر العلاقة بين دورات المحرك ودرجة حرارة الزيت",
    "7. Line Graph of Engine Load Percent & RPM": "مخطط مزدوج يظهر العلاقة بين حمل المحرك ودوراته",
    "8. Histogram of Battery Voltage": "رسم بياني يوضح توزيع قيم جهد البطارية",
    "9. Line Graph of Battery Voltage": "مخطط زمني يظهر تغيرات جهد البطارية على مدار اليوم",
    "10. Line Graph of Manifold Absolute Pressure": "مخطط ضغط الهواء داخل مشعب السحب (MAP) مقاساً بالكيلو باسكال",
    "11. Line Graph of Mass Air Flow": "مخطط زمني لتدفق كتلة الهواء (MAF) مقاساً بالجرام في الثانية",
    "12. 3D Scatter Plot of Engine Parameters": "رسم ثلاثي الأبعاد يوضح العلاقة بين دورات المحرك وتوقيت الإشعال وضغط مشعب السحب",
    "13. Line Graph of Exhaust Gas Recirculation": "مخطط زمني لحالة نظام إعادة تدوير غاز العادم (EGR)",
    "14. Line Graph of Catalytic Converter Efficiency": "مخطط زمني يوضح كفاءة عمل المحول الحفاز",
    "15. Line Graph of Brake Status": "مخطط زمني يوضح حالة الفرامل",
    "16. Line Graph of Tire Pressure": "مخطط زمني يوضح ضغط الإطارات بالـ PSI",
    "17. Line Graph of Ambient Temperature": "مخطط زمني يوضح درجة الحرارة المحيطة بالمركبة"
}

# Upload file section
st.markdown('<div class="upload-container">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h2 style='text-align: center;'>رفع ملف البيانات</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a file containing vehicle data", type="csv")

with col2:
    st.markdown("<h2 style='text-align: center;'>File info</h2>", unsafe_allow_html=True)
    if uploaded_file is not None:
        st.success("✅ uploaded successfully!")
        st.info("The file will be sent to the API for processing.")
    else:
        st.warning("⚠️ The file has not been uploaded yet!")

st.markdown('</div>', unsafe_allow_html=True)

# التحقق من رفع الملف
df = None
if uploaded_file is not None:
    with st.spinner("🔄 جاري إرسال الملف ومعالجته عبر API..."):
        try:
            # إرسال الملف إلى الـ API
            files = {'file': (uploaded_file.name, uploaded_file, 'text/csv')}
            response = requests.post(FASTAPI_URL, files=files)

            # التحقق من استجابة الـ API
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    df = pd.DataFrame(result["results"])
                    st.success("✅ تمت المعالجة والتنبؤ بنجاح!")
                    st.subheader("📋 أنواع الأعطال المحتملة الحدوث")

                    # عرض التنبؤات
                    fault_types = ['كل الأنواع'] + list(df['Predicted_Fault'].unique())
                    selected_fault = st.selectbox("اختر نوع العطل لعرضه:", fault_types)

                    if selected_fault != 'كل الأنواع':
                        filtered_df = df[df['Predicted_Fault'] == selected_fault]
                    else:
                        filtered_df = df

                    # عرض النتائج
                    if not filtered_df.empty:
                        table_data = []
                        for idx, row in filtered_df.iterrows():
                            fault_icon = {
                                'No Fault': '✅',
                                'Engine Fault': '⚠️',
                                'Electrical Fault': '⚠️',
                                'Emission Fault': '⚠️',
                                'Transmission Fault': '⚠️'
                            }.get(row['Predicted_Fault'], '❓')

                            table_data.append({
                                "Recording": f"{fault_icon} {idx + 1}",
                                "Possible fault type": row['Predicted_Fault'],
                                "Attention": row['Prediction_Message'],
                            })

                        table_df = pd.DataFrame(table_data)
                        st.dataframe(table_df, use_container_width=True)

                        # زر لتحميل النتائج كملف CSV
                        csv = table_df.to_csv(index=False)
                        st.download_button(
                            label="تحميل النتائج كملف CSV",
                            data=csv,
                            file_name="fault_predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("لا توجد نتائج لعرضها.")
                else:
                    st.error(f"❌ حدث خطأ من الخادم: {result.get('error')}")
            else:
                st.error(f"❌ حدث خطأ من الخادم: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء إرسال الملف أو المعالجة: {str(e)}")

# معالجة الرسوم البيانية فقط إذا كان df موجودًا
if df is not None:
    try:
        # تحويل الوقت إلى تنسيق التاريخ
        if 'Timestamp' in df.columns:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df['Date'] = df['Timestamp'].dt.date
            except:
                st.warning("خطأ في تحويل Timestamp إلى تنسيق التاريخ")

        st.markdown("<h2 style='text-align: center;'>--Select the charts you want to view--</h2>", unsafe_allow_html=True)

        st.markdown('<div class="chart-options-grid">', unsafe_allow_html=True)

        selected_charts = {}

        col_count = 3  # عدد الأعمدة في كل صف
        chart_items = list(chart_descriptions.items())
        rows = [chart_items[i:i+col_count] for i in range(0, len(chart_items), col_count)]

        for row in rows:
            cols = st.columns(col_count)
            for i, (chart_name, chart_desc) in enumerate(row):
                with cols[i]:
                    st.markdown(f"""
                    <div class="chart-option">
                        <h4>{chart_name}</h4>
                        <p>{chart_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    selected_charts[chart_name] = st.checkbox("عرض", key=f"chart_{chart_name}")

        st.markdown('</div>', unsafe_allow_html=True)

        if any(selected_charts.values()):
            st.markdown("<h2 style='text-align: center;'>Charts of Vehicle</h2>", unsafe_allow_html=True)

            charts_to_show = [chart for chart, selected in selected_charts.items() if selected]
            rows = [charts_to_show[i:i+2] for i in range(0, len(charts_to_show), 2)]

            for row in rows:
                cols = st.columns([1, 1])
                for i, chart_name in enumerate(row):
                    with cols[i]:
                        st.markdown(f'<div class="chart-container"><h3>{chart_name}</h3>', unsafe_allow_html=True)

                        # الرسم البياني حسب النوع
                        if chart_name == "1. Histogram of Engine RPM":
                            if 'Engine_RPM' in df.columns:
                                rpm_threshold = 6000
                                normal_rpm = df[df['Engine_RPM'] <= rpm_threshold]
                                high_rpm = df[df['Engine_RPM'] > rpm_threshold]

                                fig = go.Figure()
                                fig.add_trace(go.Histogram(
                                    x=normal_rpm['Engine_RPM'],
                                    nbinsx=50,
                                    name='دورات عادية',
                                    marker_color='green',
                                    opacity=0.75
                                ))
                                fig.add_trace(go.Histogram(
                                    x=high_rpm['Engine_RPM'],
                                    nbinsx=50,
                                    name='دورات عالية',
                                    marker_color='red',
                                    opacity=0.75
                                ))

                                fig.update_layout(
                                    title='توزيع دورات المحرك',
                                    xaxis_title='دورات المحرك (RPM)',
                                    yaxis_title='العدد',
                                    barmode='overlay',
                                    template='plotly_white',
                                    height=400,
                                    legend=dict(title='فئة الدورات')
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("عمود Engine_RPM غير موجود في البيانات")

                        elif chart_name == "2. Line Graph of Engine RPM over time":
                            if 'Engine_RPM' in df.columns and 'Timestamp' in df.columns:
                                traces = []

                                for i in range(len(df) - 1):
                                    x_segment = [df["Timestamp"].iloc[i], df["Timestamp"].iloc[i+1]]
                                    y_segment = [df["Engine_RPM"].iloc[i], df["Engine_RPM"].iloc[i+1]]

                                    color = 'orangered' if y_segment[0] > 6500 or y_segment[1] > 6500 else 'seagreen'

                                    traces.append(go.Scatter(
                                        x=x_segment,
                                        y=y_segment,
                                        mode='lines',
                                        line=dict(color=color, width=3),
                                        showlegend=False
                                    ))

                                fig = go.Figure(data=traces)
                                fig.update_layout(
                                    title="دورات المحرك عبر الزمن",
                                    xaxis_title="الوقت",
                                    yaxis_title="دورات المحرك (RPM)",
                                    xaxis=dict(
                                        tickformat="%H:%M:%S",
                                        tickangle=45
                                    ),
                                    template="plotly_white",
                                    height=400
                                )

                                fig.update_xaxes(showgrid=True, gridcolor='lightgrey')
                                fig.update_yaxes(showgrid=True, gridcolor='lightgrey')

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("بعض الأعمدة المطلوبة غير موجودة في البيانات")

                        elif chart_name == "3. Line Graph of Coolant Temperature":
                            if 'Coolant_Temp_C' in df.columns and 'Timestamp' in df.columns and 'Date' in df.columns:
                                unique_days = df['Date'].unique()

                                if len(unique_days) > 0:
                                    day = unique_days[0]  # عرض اليوم الأول فقط
                                    day_data = df[df['Date'] == day].sort_values('Timestamp')

                                    fig = go.Figure()

                                    for i in range(len(day_data) - 1):
                                        x_segment = [day_data['Timestamp'].iloc[i], day_data['Timestamp'].iloc[i+1]]
                                        y_segment = [day_data['Coolant_Temp_C'].iloc[i], day_data['Coolant_Temp_C'].iloc[i+1]]

                                        color = 'orangered' if y_segment[0] > 105 or y_segment[1] > 105 else 'seagreen'

                                        fig.add_trace(go.Scatter(
                                            x=x_segment,
                                            y=y_segment,
                                            mode='lines',
                                            line=dict(color=color, width=3),
                                            showlegend=False
                                        ))

                                    fig.add_shape(
                                        type='line',
                                        x0=day_data['Timestamp'].min(),
                                        x1=day_data['Timestamp'].max(),
                                        y0=105,
                                        y1=105,
                                        line=dict(color='red', width=2, dash='dash'),
                                    )

                                    fig.update_layout(
                                        title=f'درجة حرارة سائل التبريد بتاريخ {day}',
                                        xaxis_title='الوقت',
                                        yaxis_title='درجة حرارة سائل التبريد (°C)',
                                        xaxis=dict(tickangle=45),
                                        template='plotly_white',
                                        height=400,
                                    )

                                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error("لم يتم العثور على بيانات التاريخ")
                            else:
                                st.error("بعض الأعمدة المطلوبة غير موجودة في البيانات")

                        elif chart_name == "4. Histogram of Oil Temperature":
                            if 'Oil_Temp_C' in df.columns:
                                fig = go.Figure()

                                fig.add_trace(go.Histogram(
                                    x=df['Oil_Temp_C'],
                                    nbinsx=30,
                                    marker_color='orange',
                                    opacity=0.6,
                                    name='توزيع درجة حرارة الزيت'
                                ))

                                fig.add_shape(
                                    type='line',
                                    x0=df['Oil_Temp_C'].mean(),
                                    x1=df['Oil_Temp_C'].mean(),
                                    y0=0,
                                    y1=1,
                                    yref='paper',
                                    line=dict(color='blue', width=2, dash='dash'),
                                    name='المتوسط'
                                )

                                fig.add_shape(
                                    type='line',
                                    x0=df['Oil_Temp_C'].median(),
                                    x1=df['Oil_Temp_C'].median(),
                                    y0=0,
                                    y1=1,
                                    yref='paper',
                                    line=dict(color='green', width=2, dash='dash'),
                                    name='الوسيط'
                                )

                                fig.update_layout(
                                    title='توزيع درجة حرارة الزيت (°C)',
                                    xaxis_title='درجة حرارة الزيت (°C)',
                                    yaxis_title='التكرار',
                                    template='plotly_white',
                                    height=400,
                                    legend=dict(
                                        y=0.99,
                                        x=0.01,
                                        title_text=''
                                    )
                                )

                                fig.add_annotation(
                                    x=df['Oil_Temp_C'].mean(),
                                    y=0.95,
                                    yref='paper',
                                    text=f"المتوسط: {df['Oil_Temp_C'].mean():.1f}°C",
                                    showarrow=True,
                                    arrowhead=1,
                                    ax=50,
                                    ay=-30
                                )

                                fig.add_annotation(
                                    x=df['Oil_Temp_C'].median(),
                                    y=0.85,
                                    yref='paper',
                                    text=f"الوسيط: {df['Oil_Temp_C'].median():.1f}°C",
                                    showarrow=True,
                                    arrowhead=1,
                                    ax=-50,
                                    ay=-30
                                )

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("عمود Oil_Temp_C غير موجود في البيانات")

                        elif chart_name == "5. Line Graph of Oil Temperature":
                            if 'Oil_Temp_C' in df.columns and 'Timestamp' in df.columns and 'Date' in df.columns:
                                unique_days = df['Date'].unique()

                                if len(unique_days) > 0:
                                    day = unique_days[0]  # عرض اليوم الأول فقط
                                    day_data = df[df['Date'] == day].sort_values('Timestamp')

                                    fig = go.Figure()

                                    fig.add_trace(
                                        go.Scatter(
                                            x=day_data['Timestamp'],
                                            y=day_data['Oil_Temp_C'],
                                            mode='lines+markers',
                                            line=dict(color='darkorange', width=3),
                                            name='درجة حرارة الزيت'
                                        )
                                    )

                                    fig.update_layout(
                                        title=f'درجة حرارة الزيت (°C) بتاريخ {day}',
                                        xaxis_title='الوقت',
                                        yaxis_title='درجة حرارة الزيت (°C)',
                                        xaxis=dict(tickangle=50),
                                        template='plotly_white',
                                        height=400,
                                    )

                                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error("لم يتم العثور على بيانات التاريخ")
                            else:
                                st.error("بعض الأعمدة المطلوبة غير موجودة في البيانات")

                        elif chart_name == "6. Line Graph of Engine RPM and Oil Temperature":
                            if all(col in df.columns for col in ['Engine_RPM', 'Oil_Temp_C', 'Timestamp']):
                                fig = make_subplots(specs=[[{"secondary_y": True}]])

                                fig.add_trace(
                                    go.Scatter(
                                        x=df["Timestamp"],
                                        y=df["Engine_RPM"],
                                        mode='lines+markers',
                                        name='دورات المحرك',
                                        line=dict(color='darkgreen', width=3),
                                        marker=dict(size=7)
                                    ),
                                    secondary_y=False
                                )

                                fig.add_trace(
                                    go.Scatter(
                                        x=df["Timestamp"],
                                        y=df["Oil_Temp_C"],
                                        mode='lines+markers',
                                        name='درجة حرارة الزيت',
                                        line=dict(color='darkorange', width=3),
                                        marker=dict(size=7)
                                    ),
                                    secondary_y=True
                                )

                                fig.update_layout(
                                    title="العلاقة بين دورات المحرك ودرجة حرارة الزيت",
                                    xaxis_title="الوقت",
                                    template="plotly_white",
                                    height=400,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )

                                fig.update_yaxes(title_text="دورات المحرك (RPM)", secondary_y=False, color="darkgreen")
                                fig.update_yaxes(title_text="درجة حرارة الزيت (°C)", secondary_y=True, color="darkorange")

                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("بعض الأعمدة المطلوبة غير موجودة في البيانات")

                        elif chart_name == "7. Line Graph of Engine Load Percent & RPM":
                            if all(col in df.columns for col in ['Engine_RPM', 'Engine_Load_Percent', 'Timestamp']):
                                fig = make_subplots(specs=[[{"secondary_y": True}]])

                                fig.add_trace(
                                    go.Scatter(
                                        x=df["Timestamp"],
                                        y=df["Engine_RPM"],
                                        mode='lines',
                                        name='دورات المحرك',
                                        line=dict(color='chocolate', width=3),
                                    ),
                                    secondary_y=False
                                )

                                fig.add_trace(
                                    go.Scatter(
                                        x=df["Timestamp"],
                                        y=df["Engine_Load_Percent"],
                                        mode='lines',
                                        name='حمل المحرك (%)',
                                        line=dict(color='blue', width=3),
                                    ),
                                    secondary_y=True
                                )

                                fig.update_layout(
                                    title="العلاقة بين دورات المحرك ونسبة الحمل",
                                    xaxis_title="الوقت",
                                    template="plotly_white",
                                    height=400,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )

                                fig.update_yaxes(title_text="دورات المحرك (RPM)", secondary_y=False, color="chocolate")
                                fig.update_yaxes(title_text="نسبة حمل المحرك (%)", secondary_y=True, color="blue")

                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("بعض الأعمدة المطلوبة غير موجودة في البيانات")

                        elif chart_name == "8. Histogram of Battery Voltage":
                            if 'Battery_Voltage_V' in df.columns:
                                fig = go.Figure()

                                fig.add_trace(go.Histogram(
                                    x=df['Battery_Voltage_V'],
                                    nbinsx=30,
                                    marker_color='green',
                                    opacity=0.75,
                                    name='توزيع جهد البطارية'
                                ))

                                fig.update_layout(
                                    title="توزيع جهد البطارية",
                                    xaxis_title="جهد البطارية (فولت)",
                                    yaxis_title="التكرار",
                                    template="plotly_white",
                                    height=400
                                )

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("عمود Battery_Voltage_V غير موجود في البيانات")

                        elif chart_name == "9. Line Graph of Battery Voltage":
                            if 'Battery_Voltage_V' in df.columns and 'Timestamp' in df.columns and 'Date' in df.columns:
                                unique_days = df['Date'].unique()

                                if len(unique_days) > 0:
                                    day = unique_days[0]  # عرض اليوم الأول فقط
                                    day_data = df[df['Date'] == day].sort_values('Timestamp')

                                    fig = go.Figure()

                                    fig.add_trace(
                                        go.Scatter(
                                            x=day_data['Timestamp'],
                                            y=day_data['Battery_Voltage_V'],
                                            mode='lines+markers',
                                            line=dict(color='teal', width=3),
                                            name='جهد البطارية'
                                        )
                                    )

                                    fig.update_layout(
                                        title=f'جهد البطارية بتاريخ {day}',
                                        xaxis_title='الوقت',
                                        yaxis_title='جهد البطارية (فولت)',
                                        xaxis=dict(tickangle=50),
                                        template='plotly_white',
                                        height=400,
                                    )

                                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error("لم يتم العثور على بيانات التاريخ")
                            else:
                                st.error("بعض الأعمدة المطلوبة غير موجودة في البيانات")

                        elif chart_name == "10. Line Graph of Manifold Absolute Pressure":
                            if 'MAP_kPa' in df.columns and 'Timestamp' in df.columns:
                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    x=df['Timestamp'],
                                    y=df['MAP_kPa'],
                                    mode='lines',
                                    marker=dict(size=7, color='purple'),
                                    line=dict(width=3, color='purple'),
                                    name='ضغط مشعب السحب'
                                ))

                                fig.update_layout(
                                    title="ضغط الهواء داخل مشعب السحب (MAP_kPa)",
                                    xaxis_title="الوقت",
                                    yaxis_title="الضغط (كيلو باسكال)",
                                    template="plotly_white",
                                    height=400,
                                    xaxis=dict(tickangle=45)
                                )

                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("عمود MAP_kPa غير موجود في البيانات")

                        elif chart_name == "11. Line Graph of Mass Air Flow":
                            if 'MAF_gps' in df.columns and 'Timestamp' in df.columns:
                                fig = go.Figure()

                                fig.add_trace(go.Sc芦atter(
                                    x=df['Timestamp'],
                                    y=df['MAF_gps'],
                                    mode='lines',
                                    marker=dict(size=7, color='green'),
                                    line=dict(width=3, color='green'),
                                    name='تدفق كتلة الهواء'
                                ))

                                fig.update_layout(
                                    title="💨 تدفق كتلة الهواء (جرام/ثانية)",
                                    xaxis_title="الوقت",
                                    yaxis_title="تدفق كتلة الهواء (جرام/ثانية)",
                                    template="plotly_white",
                                    height=400,
                                    xaxis=dict(tickangle=45)
                                )

                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("عمود MAF_gps غير موجود في البيانات")

                        elif chart_name == "12. 3D Scatter Plot of Engine Parameters":
                            if all(col in df.columns for col in ['Engine_RPM', 'Ignition_Timing_Deg', 'MAP_kPa', 'MAF_gps']):
                                fig = px.scatter_3d(
                                    data_frame=df,
                                    x="Engine_RPM",
                                    y="Ignition_Timing_Deg",
                                    z="MAP_kPa",
                                    color="MAF_gps",
                                    title="عرض ثلاثي الأبعاد: دورات المحرك مقابل توقيت الإشعال مقابل ضغط المشعب (ملون حسب تدفق الهواء)",
                                    labels={
                                        "Engine_RPM": "دورات المحرك (RPM)",
                                        "Ignition_Timing_Deg": "توقيت الإشعال (درجة)",
                                        "MAP_kPa": "ضغط المشعب (كيلو باسكال)",
                                        "MAF_gps": "تدفق كتلة الهواء (جرام/ثانية)"
                                    }
                                )

                                fig.update_layout(
                                    height=600,
                                    template="plotly_white"
                                )

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("بعض الأعمدة المطلوبة غير موجودة في البيانات")

                        elif chart_name == "13. Line Graph of Exhaust Gas Recirculation":
                            if 'EGR_Status' in df.columns and 'Timestamp' in df.columns:
                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    x=df['Timestamp'],
                                    y=df['EGR_Status'],
                                    mode='lines',
                                    marker=dict(size=7, color='royalblue'),
                                    line=dict(width=3, color='royalblue'),
                                    name='حالة EGR'
                                ))

                                fig.update_layout(
                                    title="حالة نظام إعادة تدوير غاز العادم (EGR)",
                                    xaxis_title="الوقت",
                                    yaxis_title="حالة EGR (مشفرة)",
                                    template="plotly_white",
                                    height=400,
                                    xaxis=dict(tickangle=45)
                                )

                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("عمود EGR_Status غير موجود في البيانات")

                        elif chart_name == "14. Line Graph of Catalytic Converter Efficiency":
                            if 'Catalytic_Converter_Percent' in df.columns and 'Timestamp' in df.columns:
                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    x=df['Timestamp'],
                                    y=df['Catalytic_Converter_Percent'],
                                    mode='lines',
                                    marker=dict(size=7, color='teal'),
                                    line=dict(width=3, color='teal'),
                                    name='كفاءة المحول الحفاز'
                                ))

                                fig.update_layout(
                                    title="كفاءة عمل المحول الحفاز",
                                    xaxis_title="الوقت",
                                    yaxis_title="كفاءة المحول الحفاز (%)",
                                    template="plotly_white",
                                    height=400,
                                    xaxis=dict(tickangle=45)
                                )

                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("عمود Catalytic_Converter_Percent غير موجود في البيانات")

                        elif chart_name == "15. Line Graph of Brake Status":
                            if 'Brake_Status' in df.columns and 'Timestamp' in df.columns:
                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    x=df['Timestamp'],
                                    y=df['Brake_Status'],
                                    mode='lines',
                                    marker=dict(size=7, color='royalblue'),
                                    line=dict(width=3, color='royalblue'),
                                    name='حالة الفرامل'
                                ))

                                fig.update_layout(
                                    title="حالة الفرامل",
                                    xaxis_title="الوقت",
                                    yaxis_title="حالة الفرامل",
                                    template="plotly_white",
                                    height=400,
                                    xaxis=dict(tickangle=45)
                                )

                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("عمود Brake_Status غير موجود في البيانات")

                        elif chart_name == "16. Line Graph of Tire Pressure":
                            if 'Tire_Pressure_psi' in df.columns and 'Timestamp' in df.columns:
                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    x=df['Timestamp'],
                                    y=df['Tire_Pressure_psi'],
                                    mode='lines',
                                    marker=dict(size=7, color='indigo'),
                                    line=dict(width=3, color='indigo'),
                                    name='ضغط الإطارات'
                                ))

                                fig.update_layout(
                                    title="ضغط إطارات المركبة",
                                    xaxis_title="الوقت",
                                    yaxis_title="ضغط الإطارات (PSI)",
                                    template="plotly_white",
                                    height=400,
                                    xaxis=dict(tickangle=45)
                                )

                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("عمود Tire_Pressure_psi غير موجود في البيانات")

                        elif chart_name == "17. Line Graph of Ambient Temperature":
                            if 'Ambient_Temp_C' in df.columns and 'Timestamp' in df.columns:
                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    x=df['Timestamp'],
                                    y=df['Ambient_Temp_C'],
                                    mode='lines',
                                    marker=dict(size=7, color='goldenrod'),
                                    line=dict(width=3, color='goldenrod'),
                                    name='درجة الحرارة المحيطة'
                                ))

                                fig.update_layout(
                                    title="درجة الحرارة المحيطة بالمركبة",
                                    xaxis_title="الوقت",
                                    yaxis_title="درجة الحرارة (°C)",
                                    template="plotly_white",
                                    height=400,
                                    xaxis=dict(tickangle=45)
                                )

                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("عمود Ambient_Temp_C غير موجود في البيانات")

                        st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("--Choose the charts you want to view--")

    except Exception as e:
        st.error(f"خطأ: {str(e)}")

else:
    st.markdown("<div style='text-align: center; padding: 3rem;'>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #34495e; padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-top: 2rem;">
        <p>2025 | Teem OHI | Vehicle Dashboard | Vehicle Charts ©</p>
    </div>
    """, unsafe_allow_html=True)
#---------------------------------------------------------------------
