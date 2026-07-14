import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ChartGenerator:
    def __init__(self, language='ar'):
        self.language = language
        self.chart_configs = self._load_chart_configurations()
    
    def _load_chart_configurations(self):
        """Load chart configurations based on language"""
        if self.language == 'ar':
            return {
                "1. مخطط توزيع دورات المحرك": {
                    "description": "رسم بياني يوضح توزيع دورات المحرك، مع تمييز الدورات العالية والمنخفضة",
                    "type": "histogram",
                    "columns": ["Engine_RPM"],
                    "function": self._create_rpm_histogram
                },
                "2. مخطط دورات المحرك عبر الزمن": {
                    "description": "مخطط زمني يظهر تغيرات دورات المحرك مع الوقت، مع تمييز ملون للدورات العالية",
                    "type": "line",
                    "columns": ["Engine_RPM", "Timestamp"],
                    "function": self._create_rpm_timeline
                },
                "3. مخطط درجة حرارة سائل التبريد": {
                    "description": "مخطط درجة حرارة سائل التبريد على مدار الزمن، مع خط تحذير عند 105 درجة مئوية",
                    "type": "line",
                    "columns": ["Coolant_Temp_C", "Timestamp"],
                    "function": self._create_coolant_temp_chart
                },
                "4. توزيع درجة حرارة الزيت": {
                    "description": "رسم بياني يوضح توزيع درجات حرارة الزيت مع إظهار المتوسط والوسيط",
                    "type": "histogram",
                    "columns": ["Oil_Temp_C"],
                    "function": self._create_oil_temp_histogram
                },
                "5. مخطط درجة حرارة الزيت عبر الزمن": {
                    "description": "مخطط زمني يظهر تغيرات درجة حرارة الزيت على مدار اليوم",
                    "type": "line",
                    "columns": ["Oil_Temp_C", "Timestamp"],
                    "function": self._create_oil_temp_timeline
                },
                "6. العلاقة بين دورات المحرك ودرجة حرارة الزيت": {
                    "description": "مخطط مزدوج يظهر العلاقة بين دورات المحرك ودرجة حرارة الزيت",
                    "type": "dual_line",
                    "columns": ["Engine_RPM", "Oil_Temp_C", "Timestamp"],
                    "function": self._create_rpm_oil_temp_chart
                },
                "7. العلاقة بين حمل المحرك ودوراته": {
                    "description": "مخطط مزدوج يظهر العلاقة بين حمل المحرك ودوراته",
                    "type": "dual_line",
                    "columns": ["Engine_RPM", "Engine_Load_Percent", "Timestamp"],
                    "function": self._create_rpm_load_chart
                },
                "8. توزيع جهد البطارية": {
                    "description": "رسم بياني يوضح توزيع قيم جهد البطارية",
                    "type": "histogram",
                    "columns": ["Battery_Voltage_V"],
                    "function": self._create_battery_histogram
                },
                "9. مخطط جهد البطارية عبر الزمن": {
                    "description": "مخطط زمني يظهر تغيرات جهد البطارية على مدار اليوم",
                    "type": "line",
                    "columns": ["Battery_Voltage_V", "Timestamp"],
                    "function": self._create_battery_timeline
                },
                "10. ضغط مشعب السحب": {
                    "description": "مخطط ضغط الهواء داخل مشعب السحب (MAP) مقاساً بالكيلو باسكال",
                    "type": "line",
                    "columns": ["MAP_kPa", "Timestamp"],
                    "function": self._create_map_chart
                },
                "11. تدفق كتلة الهواء": {
                    "description": "مخطط زمني لتدفق كتلة الهواء (MAF) مقاساً بالجرام في الثانية",
                    "type": "line",
                    "columns": ["MAF_gps", "Timestamp"],
                    "function": self._create_maf_chart
                },
                "12. الرسم ثلاثي الأبعاد لمعاملات المحرك": {
                    "description": "رسم ثلاثي الأبعاد يوضح العلاقة بين دورات المحرك وتوقيت الإشعال وضغط مشعب السحب",
                    "type": "3d_scatter",
                    "columns": ["Engine_RPM", "Ignition_Timing_Deg", "MAP_kPa", "MAF_gps"],
                    "function": self._create_3d_scatter
                },
                "13. مخطط إعادة تدوير غاز العادم": {
                    "description": "مخطط زمني لحالة نظام إعادة تدوير غاز العادم (EGR)",
                    "type": "line",
                    "columns": ["EGR_Status", "Timestamp"],
                    "function": self._create_egr_chart
                },
                "14. مخطط كفاءة المحول الحفاز": {
                    "description": "مخطط زمني يوضح كفاءة عمل المحول الحفاز",
                    "type": "line",
                    "columns": ["Catalytic_Converter_Percent", "Timestamp"],
                    "function": self._create_catalytic_converter_chart
                },
                "15. مخطط حالة الفرامل": {
                    "description": "مخطط زمني يوضح حالة الفرامل",
                    "type": "line",
                    "columns": ["Brake_Status", "Timestamp"],
                    "function": self._create_brake_status_chart
                },
                "16. مخطط ضغط الإطارات": {
                    "description": "مخطط زمني يوضح ضغط الإطارات بالـ PSI",
                    "type": "line",
                    "columns": ["Tire_Pressure_psi", "Timestamp"],
                    "function": self._create_tire_pressure_chart
                },
                "17. مخطط درجة الحرارة المحيطة": {
                    "description": "مخطط زمني يوضح درجة الحرارة المحيطة بالمركبة",
                    "type": "line",
                    "columns": ["Ambient_Temp_C", "Timestamp"],
                    "function": self._create_ambient_temp_chart
                }
            }
        else:  # English
            return {
                "1. Engine RPM Distribution": {
                    "description": "Histogram showing engine RPM distribution with high and low RPM differentiation",
                    "type": "histogram",
                    "columns": ["Engine_RPM"],
                    "function": self._create_rpm_histogram
                },
                "2. Engine RPM Timeline": {
                    "description": "Time series showing engine RPM changes over time with color-coded high RPM periods",
                    "type": "line",
                    "columns": ["Engine_RPM", "Timestamp"],
                    "function": self._create_rpm_timeline
                },
                "3. Coolant Temperature Chart": {
                    "description": "Coolant temperature over time with warning line at 105°C",
                    "type": "line",
                    "columns": ["Coolant_Temp_C", "Timestamp"],
                    "function": self._create_coolant_temp_chart
                },
                "4. Oil Temperature Distribution": {
                    "description": "Histogram showing oil temperature distribution with mean and median indicators",
                    "type": "histogram",
                    "columns": ["Oil_Temp_C"],
                    "function": self._create_oil_temp_histogram
                },
                "5. Oil Temperature Timeline": {
                    "description": "Time series showing oil temperature changes throughout the day",
                    "type": "line",
                    "columns": ["Oil_Temp_C", "Timestamp"],
                    "function": self._create_oil_temp_timeline
                },
                "6. Engine RPM vs Oil Temperature": {
                    "description": "Dual axis chart showing relationship between engine RPM and oil temperature",
                    "type": "dual_line",
                    "columns": ["Engine_RPM", "Oil_Temp_C", "Timestamp"],
                    "function": self._create_rpm_oil_temp_chart
                },
                "7. Engine Load vs RPM": {
                    "description": "Dual axis chart showing relationship between engine load and RPM",
                    "type": "dual_line",
                    "columns": ["Engine_RPM", "Engine_Load_Percent", "Timestamp"],
                    "function": self._create_rpm_load_chart
                },
                "8. Battery Voltage Distribution": {
                    "description": "Histogram showing battery voltage value distribution",
                    "type": "histogram",
                    "columns": ["Battery_Voltage_V"],
                    "function": self._create_battery_histogram
                },
                "9. Battery Voltage Timeline": {
                    "description": "Time series showing battery voltage changes throughout the day",
                    "type": "line",
                    "columns": ["Battery_Voltage_V", "Timestamp"],
                    "function": self._create_battery_timeline
                },
                "10. Manifold Absolute Pressure": {
                    "description": "Chart showing intake manifold air pressure (MAP) in kPa",
                    "type": "line",
                    "columns": ["MAP_kPa", "Timestamp"],
                    "function": self._create_map_chart
                },
                "11. Mass Air Flow": {
                    "description": "Time series for mass air flow (MAF) in grams per second",
                    "type": "line",
                    "columns": ["MAF_gps", "Timestamp"],
                    "function": self._create_maf_chart
                },
                "12. 3D Engine Parameters Plot": {
                    "description": "3D scatter plot showing relationship between RPM, ignition timing, and MAP",
                    "type": "3d_scatter",
                    "columns": ["Engine_RPM", "Ignition_Timing_Deg", "MAP_kPa", "MAF_gps"],
                    "function": self._create_3d_scatter
                },
                "13. Exhaust Gas Recirculation Chart": {
                    "description": "Time series showing exhaust gas recirculation (EGR) status",
                    "type": "line",
                    "columns": ["EGR_Status", "Timestamp"],
                    "function": self._create_egr_chart
                },
                "14. Catalytic Converter Efficiency Chart": {
                    "description": "Time series showing catalytic converter efficiency",
                    "type": "line",
                    "columns": ["Catalytic_Converter_Percent", "Timestamp"],
                    "function": self._create_catalytic_converter_chart
                },
                "15. Brake Status Chart": {
                    "description": "Time series showing brake status",
                    "type": "line",
                    "columns": ["Brake_Status", "Timestamp"],
                    "function": self._create_brake_status_chart
                },
                "16. Tire Pressure Chart": {
                    "description": "Time series showing tire pressure in PSI",
                    "type": "line",
                    "columns": ["Tire_Pressure_psi", "Timestamp"],
                    "function": self._create_tire_pressure_chart
                },
                "17. Ambient Temperature Chart": {
                    "description": "Time series showing ambient temperature around the vehicle",
                    "type": "line",
                    "columns": ["Ambient_Temp_C", "Timestamp"],
                    "function": self._create_ambient_temp_chart
                }
            }
    
    def get_available_charts(self):
        """Return the list of available charts"""
        return self.chart_configs
    
    def create_chart(self, chart_name, data, height=500):
        """Create a chart based on the chart name and data"""
        if chart_name not in self.chart_configs:
            return {"fig": None, "description": "Chart not found"}
        
        chart_config = self.chart_configs[chart_name]
        required_columns = chart_config["columns"]
        
        # Check if all required columns exist in the data
        if not all(col in data.columns for col in required_columns):
            return {"fig": None, "description": chart_config["description"]}
        
        # Call the specific chart creation function
        fig = chart_config["function"](data, height)
        return {"fig": fig, "description": chart_config["description"]}
    
    def _create_rpm_histogram(self, data, height):
        """Create histogram for engine RPM distribution"""
        labels = {
            "ar": {"title": "توزيع دورات المحرك", "x": "دورات المحرك ( RPM )", "y": "العدد"},
            "en": {"title": "Engine RPM Distribution", "x": "Engine RPM", "y": "Count"}
        }
        
        fig = px.histogram(
            data,
            x="Engine_RPM",
            nbins=50,
            title=labels[self.language]["title"],
            color_discrete_sequence=["#3498db"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12),
            showlegend=False
        )
        
        # Add high RPM threshold line
        fig.add_vline(
            x=4000,
            line_dash="dash",
            line_color="red",
            annotation_text="High RPM" if self.language == "en" else "دورات عالية",
            annotation_position="top"
        )
        
        return fig
    
    def _create_rpm_timeline(self, data, height):
        """Create line chart for engine RPM over time"""
        labels = {
            "ar": {"title": "دورات المحرك عبر الزمن", "x": "الوقت", "y": "دورات المحرك (RPM)"},
            "en": {"title": "Engine RPM Timeline", "x": "Time", "y": "Engine RPM"}
        }
        
        fig = px.line(
            data,
            x="Timestamp",
            y="Engine_RPM",
            title=labels[self.language]["title"],
            color_discrete_sequence=["#3498db"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12)
        )
        
        # Add high RPM threshold line
        fig.add_hline(
            y=4000,
            line_dash="dash",
            line_color="red",
            annotation_text="High RPM" if self.language == "en" else "دورات عالية",
            annotation_position="top right"
        )
        
        return fig
    
    def _create_coolant_temp_chart(self, data, height):
        """Create line chart for coolant temperature over time"""
        labels = {
            "ar": {"title": "درجة حرارة سائل التبريد", "x": "الوقت", "y": "درجة الحرارة (°C)"},
            "en": {"title": "Coolant Temperature", "x": "Time", "y": "Temperature (°C)"}
        }
        
        fig = px.line(
            data,
            x="Timestamp",
            y="Coolant_Temp_C",
            title=labels[self.language]["title"],
            color_discrete_sequence=["#e74c3c"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12)
        )
        
        # Add warning line at 105°C
        fig.add_hline(
            y=105,
            line_dash="dash",
            line_color="red",
            annotation_text="Warning" if self.language == "en" else "تحذير",
            annotation_position="top right"
        )
        
        return fig
    
    def _create_oil_temp_histogram(self, data, height):
        """Create histogram for oil temperature distribution"""
        labels = {
            "ar": {"title": "توزيع درجة حرارة الزيت", "x": "درجة الحرارة (°C)", "y": "العدد"},
            "en": {"title": "Oil Temperature Distribution", "x": "Temperature (°C)", "y": "Count"}
        }
        
        fig = px.histogram(
            data,
            x="Oil_Temp_C",
            nbins=50,
            title=labels[self.language]["title"],
            color_discrete_sequence=["#e67e22"]
        )
        
        # Add mean and median lines
        mean_temp = data["Oil_Temp_C"].mean()
        median_temp = data["Oil_Temp_C"].median()
        
        fig.add_vline(
            x=mean_temp,
            line_dash="dash",
            line_color="blue",
            annotation_text="Mean" if self.language == "en" else "المتوسط",
            annotation_position="top left"
        )
        fig.add_vline(
            x=median_temp,
            line_dash="dash",
            line_color="green",
            annotation_text="Median" if self.language == "en" else "الوسيط",
            annotation_position="top right"
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12),
            showlegend=False
        )
        
        return fig
    
    def _create_oil_temp_timeline(self, data, height):
        """Create line chart for oil temperature over time"""
        labels = {
            "ar": {"title": "درجة حرارة الزيت عبر الزمن", "x": "الوقت", "y": "درجة الحرارة (°C)"},
            "en": {"title": "Oil Temperature Timeline", "x": "Time", "y": "Temperature (°C)"}
        }
        
        fig = px.line(
            data,
            x="Timestamp",
            y="Oil_Temp_C",
            title=labels[self.language]["title"],
            color_discrete_sequence=["#e67e22"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12)
        )
        
        return fig
    
    def _create_rpm_oil_temp_chart(self, data, height):
        """Create dual axis chart for engine RPM and oil temperature"""
        labels = {
            "ar": {
                "title": "دورات المحرك مقابل درجة حرارة الزيت",
                "x": "الوقت",
                "y1": "دورات المحرك (RPM)",
                "y2": "درجة الحرارة (°C)"
            },
            "en": {
                "title": "Engine RPM vs Oil Temperature",
                "x": "Time",
                "y1": "Engine RPM",
                "y2": "Temperature (°C)"
            }
        }
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Engine RPM trace
        fig.add_trace(
            go.Scatter(
                x=data["Timestamp"],
                y=data["Engine_RPM"],
                name="RPM" if self.language == "en" else "دورات المحرك",
                line=dict(color="#3498db")
            ),
            secondary_y=False
        )
        
        # Oil temperature trace
        fig.add_trace(
            go.Scatter(
                x=data["Timestamp"],
                y=data["Oil_Temp_C"],
                name="Oil Temp" if self.language == "en" else "درجة حرارة الزيت",
                line=dict(color="#e67e22")
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title=labels[self.language]["title"],
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y1"],
            yaxis2_title=labels[self.language]["y2"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12)
        )
        
        return fig
    
    def _create_rpm_load_chart(self, data, height):
        """Create dual axis chart for engine RPM and load"""
        labels = {
            "ar": {
                "title": "دورات المحرك مقابل حمل المحرك",
                "x": "الوقت",
                "y1": "دورات المحرك (RPM)",
                "y2": "حمل المحرك (%)"
            },
            "en": {
                "title": "Engine RPM vs Engine Load",
                "x": "Time",
                "y1": "Engine RPM",
                "y2": "Engine Load (%)"
            }
        }
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Engine RPM trace
        fig.add_trace(
            go.Scatter(
                x=data["Timestamp"],
                y=data["Engine_RPM"],
                name="RPM" if self.language == "en" else "دورات المحرك",
                line=dict(color="#3498db")
            ),
            secondary_y=False
        )
        
        # Engine load trace
        fig.add_trace(
            go.Scatter(
                x=data["Timestamp"],
                y=data["Engine_Load_Percent"],
                name="Load" if self.language == "en" else "حمل المحرك",
                line=dict(color="#27ae60")
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title=labels[self.language]["title"],
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y1"],
            yaxis2_title=labels[self.language]["y2"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12)
        )
        
        return fig
    
    def _create_battery_histogram(self, data, height):
        """Create histogram for battery voltage distribution"""
        labels = {
            "ar": {"title": "توزيع جهد البطارية", "x": "الجهد (V)", "y": "العدد"},
            "en": {"title": "Battery Voltage Distribution", "x": "Voltage (V)", "y": "Count"}
        }
        
        fig = px.histogram(
            data,
            x="Battery_Voltage_V",
            nbins=50,
            title=labels[self.language]["title"],
            color_discrete_sequence=["#27ae60"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12),
            showlegend=False
        )
        
        return fig
    
    def _create_battery_timeline(self, data, height):
        """Create line chart for battery voltage over time"""
        labels = {
            "ar": {"title": "جهد البطارية عبر الزمن", "x": "الوقت", "y": "الجهد (V)"},
            "en": {"title": "Battery Voltage Timeline", "x": "Time", "y": "Voltage (V)"}
        }
        
        fig = px.line(
            data,
            x="Timestamp",
            y="Battery_Voltage_V",
            title=labels[self.language]["title"],
            color_discrete_sequence=["#27ae60"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12)
        )
        
        # Add warning line at 11.5V
        fig.add_hline(
            y=11.5,
            line_dash="dash",
            line_color="red",
            annotation_text="Low Voltage" if self.language == "en" else "جهد منخفض",
            annotation_position="top right"
        )
        
        return fig
    
    def _create_map_chart(self, data, height):
        """Create line chart for manifold absolute pressure"""
        labels = {
            "ar": {"title": "ضغط مشعب السحب", "x": "الوقت", "y": "الضغط (kPa)"},
            "en": {"title": "Manifold Absolute Pressure", "x": "Time", "y": "Pressure (kPa)"}
        }
        
        fig = px.line(
            data,
            x="Timestamp",
            y="MAP_kPa",
            title=labels[self.language]["title"],
            color_discrete_sequence=["#9b59b6"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12)
        )
        
        return fig
    
    def _create_maf_chart(self, data, height):
        """Create line chart for mass air flow"""
        labels = {
            "ar": {"title": "تدفق كتلة الهواء", "x": "الوقت", "y": "التدفق (g/s)"},
            "en": {"title": "Mass Air Flow", "x": "Time", "y": "Flow (g/s)"}
        }
        
        # Ensure data is sorted by Timestamp
        data = data.sort_values("Timestamp")
        
        fig = px.line(
            data,
            x="Timestamp",
            y="MAF_gps",
            title=labels[self.language]["title"],
            color_discrete_sequence=["#2ecc71"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12),
            xaxis=dict(
                tickangle=45,
                tickformat="%H:%M:%S"
            )
        )
        
        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        
        return fig
    
    def _create_3d_scatter(self, data, height):
        """Create 3D scatter plot for engine parameters"""
        labels = {
            "ar": {
                "title": "العلاقة ثلاثية الأبعاد لمعاملات المحرك",
                "x": "دورات المحرك (RPM)",
                "y": "توقيت الإشعال (°)",
                "z": "ضغط مشعب السحب (kPa)"
            },
            "en": {
                "title": "3D Engine Parameters",
                "x": "Engine RPM",
                "y": "Ignition Timing (°)",
                "z": "MAP (kPa)"
            }
        }
        
        fig = px.scatter_3d(
            data,
            x="Engine_RPM",
            y="Ignition_Timing_Deg",
            z="MAP_kPa",
            color="MAF_gps",
            title=labels[self.language]["title"],
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=labels[self.language]["x"],
                yaxis_title=labels[self.language]["y"],
                zaxis_title=labels[self.language]["z"]
            ),
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12)
        )
        
        return fig
    
    def _create_egr_chart(self, data, height):
        """Create line chart for exhaust gas recirculation (EGR) status"""
        labels = {
            "ar": {"title": "حالة نظام إعادة تدوير غاز العادم (EGR)", "x": "الوقت", "y": "حالة EGR (مشفرة)"},
            "en": {"title": "Exhaust Gas Recirculation (EGR) Status", "x": "Time", "y": "EGR Status (Encoded)"}
        }
        
        fig = px.line(
            data,
            x="Timestamp",
            y="EGR_Status",
            title=labels[self.language]["title"],
            color_discrete_sequence=["#2980b9"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12),
            xaxis=dict(
                tickangle=45,
                tickformat="%H:%M:%S"
            )
        )
        
        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        
        return fig
    
    def _create_catalytic_converter_chart(self, data, height):
        """Create line chart for catalytic converter efficiency"""
        labels = {
            "ar": {"title": "كفاءة المحول الحفاز", "x": "الوقت", "y": "كفاءة المحول الحفاز (%)"},
            "en": {"title": "Catalytic Converter Efficiency", "x": "Time", "y": "Catalytic Converter Efficiency (%)"}
        }
        
        fig = px.line(
            data,
            x="Timestamp",
            y="Catalytic_Converter_Percent",
            title=labels[self.language]["title"],
            color_discrete_sequence=["#16a085"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12),
            xaxis=dict(
                tickangle=45,
                tickformat="%H:%M:%S"
            )
        )
        
        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        
        return fig
    
    def _create_brake_status_chart(self, data, height):
        """Create line chart for brake status"""
        labels = {
            "ar": {"title": "حالة الفرامل", "x": "الوقت", "y": "حالة الفرامل"},
            "en": {"title": "Brake Status", "x": "Time", "y": "Brake Status"}
        }
        
        fig = px.line(
            data,
            x="Timestamp",
            y="Brake_Status",
            title=labels[self.language]["title"],
            color_discrete_sequence=["#2980b9"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12),
            xaxis=dict(
                tickangle=45,
                tickformat="%H:%M:%S"
            )
        )
        
        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        
        return fig
    
    def _create_tire_pressure_chart(self, data, height):
        """Create line chart for tire pressure"""
        labels = {
            "ar": {"title": "ضغط إطارات المركبة", "x": "الوقت", "y": "ضغط الإطارات (PSI)"},
            "en": {"title": "Tire Pressure", "x": "Time", "y": "Tire Pressure (PSI)"}
        }
        
        fig = px.line(
            data,
            x="Timestamp",
            y="Tire_Pressure_psi",
            title=labels[self.language]["title"],
            color_discrete_sequence=["#8e44ad"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12),
            xaxis=dict(
                tickangle=45,
                tickformat="%H:%M:%S"
            )
        )
        
        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        
        return fig
    
    def _create_ambient_temp_chart(self, data, height):
        """Create line chart for ambient temperature"""
        labels = {
            "ar": {"title": "درجة الحرارة المحيطة بالمركبة", "x": "الوقت", "y": "درجة الحرارة (°C)"},
            "en": {"title": "Ambient Temperature", "x": "Time", "y": "Temperature (°C)"}
        }
        
        fig = px.line(
            data,
            x="Timestamp",
            y="Ambient_Temp_C",
            title=labels[self.language]["title"],
            color_discrete_sequence=["#f1c40f"]
        )
        
        fig.update_layout(
            xaxis_title=labels[self.language]["x"],
            yaxis_title=labels[self.language]["y"],
            height=height,
            template="plotly_white",
            font=dict(family="Cairo" if self.language == "ar" else "Arial", size=12),
            xaxis=dict(
                tickangle=45,
                tickformat="%H:%M:%S"
            )
        )
        
        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        
        return fig