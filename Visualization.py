import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from io import StringIO
from plotly.subplots import make_subplots


# streamlit configuration
st.set_page_config(
    page_title="Vehicle Dashboard",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .header-container {
        background-color: #34495e;
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .upload-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .chart-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
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
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .chart-option p {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    .selected {
        border: 2px solid #3498db;
        background-color: #ebf5fb;
    }
    .chart-options-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="header-container"><h1> Vehicle Data Plate ⚙ </h1></div>', unsafe_allow_html=True)

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

# upload file section
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h2 style='text-align: center;'>Upload data file</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file containing vehicle data.", type="csv")

with col2:
    st.markdown("<h2 style='text-align: center;'> File info </h2>", unsafe_allow_html=True)
    if uploaded_file is not None:
        st.success("✅ The file has been uploaded!")
        st.info("You can now choose which charts you want to view from the options below.")
    else:
        st.warning("⚠️Failed to upload the file")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        if 'Timestamp' in data.columns:
            try:
                data['Timestamp'] = pd.to_datetime(data['Timestamp'])
                data['Date'] = data['Timestamp'].dt.date
            except:
                st.warning("error converting Timestamp to datetime format")
        
        st.markdown("<h2 style='text-align: center;'> Choose the charts you want to view </h2>", unsafe_allow_html=True)

        st.markdown('<div class="chart-options-grid">', unsafe_allow_html=True)
        
        selected_charts = {}
        
        col_count = 3  
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
                    selected_charts[chart_name] = st.checkbox("show", key=f"chart_{chart_name}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if any(selected_charts.values()):
            st.markdown("<h2 style='text-align: center;'>Vehicle Charts</h2>", unsafe_allow_html=True)

            charts_to_show = [chart for chart, selected in selected_charts.items() if selected]
            rows = [charts_to_show[i:i+2] for i in range(0, len(charts_to_show), 2)]
            
            for row in rows:
                cols = st.columns([1, 1])
                for i, chart_name in enumerate(row):
                    with cols[i]:
                        st.markdown(f'<div class="chart-container"><h3>{chart_name}</h3>', unsafe_allow_html=True)
                        
##--------------------------------------------------------------  
## Chart by type
                        if chart_name == "1. Histogram of Engine RPM":
                            if 'Engine_RPM' in data.columns:
                                rpm_threshold = 6000
                                normal_rpm = data[data['Engine_RPM'] <= rpm_threshold]
                                high_rpm = data[data['Engine_RPM'] > rpm_threshold]
                                
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
                            if 'Engine_RPM' in data.columns and 'Timestamp' in data.columns:
                                traces = []
                                
                                for i in range(len(data) - 1):
                                    x_segment = [data["Timestamp"].iloc[i], data["Timestamp"].iloc[i+1]]
                                    y_segment = [data["Engine_RPM"].iloc[i], data["Engine_RPM"].iloc[i+1]]
                                    
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
                            if 'Coolant_Temp_C' in data.columns and 'Timestamp' in data.columns and 'Date' in data.columns:
                                unique_days = data['Date'].unique()
                                
                                if len(unique_days) > 0:
                                    day = unique_days[0] 
                                    day_data = data[data['Date'] == day].sort_values('Timestamp')
                                    
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
                            if 'Oil_Temp_C' in data.columns:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Histogram(
                                    x=data['Oil_Temp_C'],
                                    nbinsx=30,
                                    marker_color='orange',
                                    opacity=0.6,
                                    name='توزيع درجة حرارة الزيت'
                                ))
                                
                                fig.add_shape(
                                    type='line',
                                    x0=data['Oil_Temp_C'].mean(),
                                    x1=data['Oil_Temp_C'].mean(),
                                    y0=0,
                                    y1=1,
                                    yref='paper',
                                    line=dict(color='blue', width=2, dash='dash'),
                                    name='المتوسط'
                                )
                                
                                fig.add_shape(
                                    type='line',
                                    x0=data['Oil_Temp_C'].median(),
                                    x1=data['Oil_Temp_C'].median(),
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
                                    x=data['Oil_Temp_C'].mean(),
                                    y=0.95,
                                    yref='paper',
                                    text=f"المتوسط: {data['Oil_Temp_C'].mean():.1f}°C",
                                    showarrow=True,
                                    arrowhead=1,
                                    ax=50,
                                    ay=-30
                                )
                                
                                fig.add_annotation(
                                    x=data['Oil_Temp_C'].median(),
                                    y=0.85,
                                    yref='paper',
                                    text=f"الوسيط: {data['Oil_Temp_C'].median():.1f}°C",
                                    showarrow=True,
                                    arrowhead=1,
                                    ax=-50,
                                    ay=-30
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("عمود Oil_Temp_C غير موجود في البيانات")
                        
                        elif chart_name == "5. Line Graph of Oil Temperature":
                            if 'Oil_Temp_C' in data.columns and 'Timestamp' in data.columns and 'Date' in data.columns:
                                unique_days = data['Date'].unique()
                                
                                if len(unique_days) > 0:
                                    day = unique_days[0] 
                                    day_data = data[data['Date'] == day].sort_values('Timestamp')
                                    
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
                            if all(col in data.columns for col in ['Engine_RPM', 'Oil_Temp_C', 'Timestamp']):
                                fig = make_subplots(specs=[[{"secondary_y": True}]])
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=data["Timestamp"],
                                        y=data["Engine_RPM"],
                                        mode='lines+markers',
                                        name='دورات المحرك',
                                        line=dict(color='darkgreen', width=3),
                                        marker=dict(size=7)
                                    ),
                                    secondary_y=False
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=data["Timestamp"],
                                        y=data["Oil_Temp_C"],
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
                            if all(col in data.columns for col in ['Engine_RPM', 'Engine_Load_Percent', 'Timestamp']):
                                fig = make_subplots(specs=[[{"secondary_y": True}]])
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=data["Timestamp"],
                                        y=data["Engine_RPM"],
                                        mode='lines',
                                        name='دورات المحرك',
                                        line=dict(color='chocolate', width=3),
                                    ),
                                    secondary_y=False
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=data["Timestamp"],
                                        y=data["Engine_Load_Percent"],
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
                            if 'Battery_Voltage_V' in data.columns:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Histogram(
                                    x=data['Battery_Voltage_V'],
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
                            if 'Battery_Voltage_V' in data.columns and 'Timestamp' in data.columns and 'Date' in data.columns:
                                unique_days = data['Date'].unique()
                                
                                if len(unique_days) > 0:
                                    day = unique_days[0]  # عرض اليوم الأول فقط
                                    day_data = data[data['Date'] == day].sort_values('Timestamp')
                                    
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
                            if 'MAP_kPa' in data.columns and 'Timestamp' in data.columns:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=data['Timestamp'],
                                    y=data['MAP_kPa'],
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
                            if 'MAF_gps' in data.columns and 'Timestamp' in data.columns:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=data['Timestamp'],
                                    y=data['MAF_gps'],
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
                            if all(col in data.columns for col in ['Engine_RPM', 'Ignition_Timing_Deg', 'MAP_kPa', 'MAF_gps']):
                                fig = px.scatter_3d(
                                    data_frame=data,
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
                            if 'EGR_Status' in data.columns and 'Timestamp' in data.columns:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=data['Timestamp'],
                                    y=data['EGR_Status'],
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
                            if 'Catalytic_Converter_Percent' in data.columns and 'Timestamp' in data.columns:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=data['Timestamp'],
                                    y=data['Catalytic_Converter_Percent'],
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
                            if 'Brake_Status' in data.columns and 'Timestamp' in data.columns:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=data['Timestamp'],
                                    y=data['Brake_Status'],
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
                            if 'Tire_Pressure_psi' in data.columns and 'Timestamp' in data.columns:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=data['Timestamp'],
                                    y=data['Tire_Pressure_psi'],
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
                            if 'Ambient_Temp_C' in data.columns and 'Timestamp' in data.columns:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=data['Timestamp'],
                                    y=data['Ambient_Temp_C'],
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
        st.error(f"Error: {str(e)}")
else:
    st.markdown("<div style='text-align: center; padding: 3rem;'>", unsafe_allow_html=True)
    st.markdown("###  To display the dashboard")
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("""
<div style="background-color: #34495e; padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-top: 2rem;">
    <p> 2025 | Teem OHI |Vehicle Dashboard | Vehical Charts ©</p>
</div>
""", unsafe_allow_html=True)
