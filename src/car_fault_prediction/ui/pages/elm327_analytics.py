import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

TRANSLATIONS = {
    'ar': {
        'analytics_title': 'إحصائيات البيانات',
        'total_records': 'إجمالي السجلات',
        'fault_percentage': 'نسبة الأعطال',
        'avg_value': 'المتوسط',
        'min_value': 'الحد الأدنى',
        'max_value': 'الحد الأقصى',
        'std_dev': 'الانحراف المعياري',
        'no_data': 'لا توجد بيانات متاحة',
        'data_with_predictions': 'البيانات مع التنبؤات',
        'download_results': 'تحميل النتائج كملف CSV',
        'stats_overview': 'نظرة عامة على الإحصائيات',
        'distribution_analysis': 'تحليل التوزيع'
    },
    'en': {
        'analytics_title': 'Data Analytics',
        'total_records': 'Total Records',
        'fault_percentage': 'Fault Percentage',
        'avg_value': 'Average',
        'min_value': 'Min',
        'max_value': 'Max',
        'std_dev': 'Std Dev',
        'no_data': 'No data available',
        'data_with_predictions': 'Data with Predictions',
        'download_results': 'Download Results as CSV',
        'stats_overview': 'Statistics Overview',
        'distribution_analysis': 'Distribution Analysis'
    }
}

IMPORTANT_COLUMNS = [
    'Coolant_Temp_C', 'Oil_Temp_C', 'Vehical_Speed_kmh',
    'Battery_Voltage_V', 'Catalytic_Converter_Percent',
    'Charging_System_Status', 'O2_Sensor_V'
]

def categorize_value(column, value):
    categories = {
        "Coolant_Temp_C": {
            (0, 70): "🔵 منخفض",
            (70, 90): "🟢 طبيعي", 
            (90, 105): "🟡 مرتفع",
            (105, float('inf')): "🔴 خطورة"
        },
        "Oil_Temp_C": {
            (0, 60): "🔵 منخفض",
            (60, 95): "🟢 طبيعي",
            (95, 110): "🟡 مرتفع", 
            (110, float('inf')): "🔴 خطورة"
        },
        "Catalytic_Converter_Percent": {
            (0, 25): "🟢 ممتاز",
            (25, 50): "🟡 جيد",
            (50, 75): "🟠 ضعيف",
            (75, float('inf')): "🔴 خطورة"
        },
        "Battery_Voltage_V": {
            (0, 11.5): "🔴 منخفض",
            (11.5, 12.2): "🟠 ضعيف",
            (12.2, 13.5): "🟢 طبيعي",
            (13.5, float('inf')): "⚡ مرتفع"
        },
        "Vehical_Speed_kmh": {
            (0, 0.1): "🛑 موقف",
            (0.1, 80): "🟢 قيادة طبيعية",
            (80, 120): "🟠 سريع",
            (120, float('inf')): "🔴 مفرط"
        },
        "O2_Sensor_V": lambda x: "🔴 غير طبيعي" if x < 0.1 or x > 0.9 else "🟢 طبيعي"
    }
    
    if column not in categories:
        return "غير معروف"
    
    cat_rules = categories[column]
    if callable(cat_rules):
        return cat_rules(value)
    
    for (min_val, max_val), label in cat_rules.items():
        if min_val <= value < max_val:
            return label
    return "غير معروف"

def create_column_analysis(data, column):
    
    if column not in data.columns or data[column].empty:
        return None, None
    
    col_data = data[column].dropna()
    
    stats = {}
    if pd.api.types.is_numeric_dtype(col_data):
        stats = {
            'الحد الأدنى': col_data.min(),
            'الحد الأقصى': col_data.max(), 
            'المتوسط': col_data.mean(),
            'الوسيط': col_data.median(),
            'الانحراف المعياري': col_data.std()
        }
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['توزيع القيم', 'توزيع الفئات'],
        specs=[[{"secondary_y": False}, {"type": "pie"}]]
    )
    
    #  Histogram - Box Plot
    if pd.api.types.is_numeric_dtype(col_data):
        fig.add_trace(
            go.Histogram(x=col_data, name='توزيع القيم', nbinsx=20),
            row=1, col=1
        )
    
    if column in ["Coolant_Temp_C", "Oil_Temp_C", "Catalytic_Converter_Percent", "Battery_Voltage_V", "Vehical_Speed_kmh", "O2_Sensor_V"]:
        categorized = col_data.apply(lambda x: categorize_value(column, x))
        value_counts = categorized.value_counts()
        
        colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
        
        fig.add_trace(
            go.Pie(
                labels=value_counts.index,
                values=value_counts.values,
                hole=0.3,
                marker_colors=colors[:len(value_counts)]
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text=f"Analysis {column}",
        height=350,
        showlegend=True
    )
    
    return stats, fig

def display_analytics_page():
    t = TRANSLATIONS[st.session_state.get('lang_code', 'ar')]
    st.markdown(f"""
    <div style='border: 2px solid #007BFF; border-radius: 12px; padding: 16px 0 16px 0; background: #181c24; margin-bottom: 18px;'>
        <h2 style='color: #007BFF; text-align: center; margin: 0;'>📊 {t['analytics_title']}</h2>
    </div>
    """, unsafe_allow_html=True)

    st_autorefresh(interval=5000, key="analytics_refresh")

    if 'simulated_data' not in st.session_state or st.session_state['simulated_data'].empty:
        st.markdown(f"<p style='text-align: center;'>{t['no_data']}</p>", unsafe_allow_html=True)
        return

    if 'predictions' in st.session_state and not st.session_state['predictions'].empty:
        merged_data = st.session_state['simulated_data'].merge(
            st.session_state['predictions'][['Predicted_Fault']],
            left_index=True, right_index=True, how='left'
        )
    else:
        merged_data = st.session_state['simulated_data'].copy()

    total_rows = len(merged_data)
    fault_count = len(merged_data[merged_data['Status'] == 'Fault'])
    fault_percentage = (fault_count / total_rows * 100) if total_rows > 0 else 0


    st.markdown("""
    <div style='border: 2px solid #444; border-radius: 12px; padding: 18px; background: #181c24; margin-bottom: 10px;'>
        <div style='display: flex; justify-content: space-between;'>
            <div style='flex:1; text-align:center;'>
                <div style='font-size: 18px; color: #aaa;'>{}</div>
                <div style='font-size: 28px; font-weight: bold; color: #fff;'>{}</div>
            </div>
            <div style='flex:1; text-align:center;'>
                <div style='font-size: 18px; color: #aaa;'>{}</div>
                <div style='font-size: 28px; font-weight: bold; color: #fff;'>{}</div>
            </div>
            <div style='flex:1; text-align:center;'>
                <div style='font-size: 18px; color: #aaa;'>عدد الأعطال</div>
                <div style='font-size: 28px; font-weight: bold; color: #fff;'>{}</div>
            </div>
        </div>
    </div>
    """.format(
        t['total_records'], total_rows,
        t['fault_percentage'], f"{fault_percentage:.1f}%",
        fault_count
    ), unsafe_allow_html=True)

    fault_labels = ['Normal', 'Fault']
    fault_values = [total_rows - fault_count, fault_count]
    fig_fault = px.pie(
        names=fault_labels, values=fault_values,
        title="نسبة الأعطال الإجمالية",
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    st.plotly_chart(fig_fault, use_container_width=True)
    st.divider()

    st.markdown(f"### {t['distribution_analysis']}")
    
    available_columns = [col for col in IMPORTANT_COLUMNS if col in merged_data.columns]
    
    for i in range(0, len(available_columns), 2):
        col1, col2 = st.columns(2)
        
        # First column in the pair
        if i < len(available_columns):
            column1 = available_columns[i]
            with col1:
                with st.container():
                    st.markdown(f"""
                    <div style='border: 2px solid #007BFF; border-radius: 10px; padding: 8px 0 8px 0; background: #181c24; margin-bottom: 8px;'>
                        <h4 style='color: #007BFF; text-align: center; margin: 0;'>{column1}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    stats1, fig1 = create_column_analysis(merged_data, column1)
                    

                    if stats1:
                        st.markdown("الإحصائيات:")
                        stats_col1, stats_col2 = st.columns(2)
                        with stats_col1:
                            st.metric("الحد الأدنى", f"{stats1['الحد الأدنى']:.2f}")
                            st.metric("المتوسط", f"{stats1['المتوسط']:.2f}")
                        with stats_col2:
                            st.metric("الحد الأقصى", f"{stats1['الحد الأقصى']:.2f}")
                            st.metric("الوسيط", f"{stats1['الوسيط']:.2f}")
                    
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True, key=f"chart_{column1}")
                        
        # Second column in the pair
        if i + 1 < len(available_columns):
            column2 = available_columns[i + 1]
            with col2:
                with st.container():
                    st.markdown(f"""
                    <div style='border: 2px solid #007BFF; border-radius: 10px; padding: 8px 0 8px 0; background: #181c24; margin-bottom: 8px;'>
                        <h4 style='color: #007BFF; text-align: center; margin: 0;'>{column2}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    stats2, fig2 = create_column_analysis(merged_data, column2)
                    

                    if stats2:
                        st.markdown("الإحصائيات:")
                        stats_col1, stats_col2 = st.columns(2)
                        with stats_col1:
                            st.metric("الحد الأدنى", f"{stats2['الحد الأدنى']:.2f}")
                            st.metric("المتوسط", f"{stats2['المتوسط']:.2f}")
                        with stats_col2:
                            st.metric("الحد الأقصى", f"{stats2['الحد الأقصى']:.2f}")
                            st.metric("الوسيط", f"{stats2['الوسيط']:.2f}")
                    
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True, key=f"chart_{column2}")
        
        if i + 2 < len(available_columns):
            st.divider()

    st.markdown(f"<h3 style='color: #007BFF;'>{t['data_with_predictions']}</h3>", unsafe_allow_html=True)
    st.dataframe(merged_data.tail(10), use_container_width=True)

    csv = merged_data.to_csv(index=False)
    st.download_button(
        label=t['download_results'],
        data=csv,
        file_name="data_with_predictions.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    display_analytics_page()
