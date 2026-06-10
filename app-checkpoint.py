# =============================================================
#  PUNE NO2 AIR QUALITY MONITOR — Streamlit Interface
#
#  Install:
#    pip install streamlit plotly pandas numpy
#
#  Run:
#    streamlit run app.py
#
#  Data files needed (put in same folder as app.py):
#    data/output/all_predictions_500m.csv   ← full pixel predictions
#    data/output/training_data_500m.csv     ← for station comparison
#    OR (fallback):
#    merged_full.csv
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Pune NO₂ Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CONSTANTS ──────────────────────────────────────────────────
WHO_LIMIT   = 25.0
NAAQS_LIMIT = 80.0

SEASON_NAMES  = {0: "Winter", 1: "Summer", 2: "Monsoon", 3: "Post-monsoon"}
SEASON_COLORS = {
    "Winter":       "#3B8BD4",
    "Summer":       "#EF9F27",
    "Monsoon":      "#1D9E75",
    "Post-monsoon": "#D4537E",
}

AQI_LEVELS = [
    (0,   40,  "#00E400", "Good",          "Air quality is satisfactory. No health risk."),
    (40,  80,  "#FFFF00", "Satisfactory",  "Acceptable for most people. Sensitive groups take caution."),
    (80,  120, "#FF7E00", "Moderate",      "Exceeds India NAAQS. May cause breathing discomfort."),
    (120, 180, "#FF0000", "Poor",          "Health effects possible for all groups."),
    (180, 250, "#8F3F97", "Very Poor",     "Serious risk. Avoid outdoor activity."),
    (250, 999, "#7E0023", "Severe",        "Emergency conditions. Stay indoors."),
]

NO2_COLORSCALE = [
    [0.00, "#00E400"], [0.20, "#FFFF00"],
    [0.40, "#FF7E00"], [0.60, "#FF0000"],
    [0.80, "#8F3F97"], [1.00, "#7E0023"],
]

STATIONS = {
    "Bhumkar Nagar, Pune - IITM":                  (18.598,   73.773),
    "Gavalinagar_Pimpri_Chinchwad":                (18.63673, 73.82487),
    "Hadapsar, Pune - IITM":                       (18.503,   73.939),
    "Katraj_Dairy_Pune":                           (18.45445, 73.85416),
    "Panchawati_Pashan, Pune - IITM":              (18.536,   73.826),
    "Park Street Wakad, Pimpri Chinchwad - MPCB":  (18.597,   73.762),
    "Savitribai Phule Pune University, Pune - MPCB":(18.529,  73.851),
    "Savta Mali Nagar, Pimpri-Chinchwad - IITM":   (18.628,   73.806),
    "Thergaon, Pimpri Chinchwad - MPCB":           (18.611,   73.784),
    "Transport Nagar-Nigdi, Pune - IITM":          (18.654,   73.788),
}

PRED_PATHS = [
    "data/output/all_predictions_500m.csv",
    "map_outputs/all_predictions.csv",
    "data/output/training_data_500m.csv",
    "merged_full.csv",
]

TRAIN_PATHS = [
    "data/output/training_data_500m.csv",
    "merged_full.csv",
]


# ── CSS STYLING ────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 4px solid #2196F3;
    margin-bottom: 10px;
  }
  .metric-label { font-size: 12px; color: #666; margin-bottom: 2px; }
  .metric-value { font-size: 24px; font-weight: 600; color: #1a1a1a; }
  .metric-sub   { font-size: 11px; color: #888; margin-top: 2px; }

  .aqi-banner {
    border-radius: 8px;
    padding: 12px 20px;
    margin: 8px 0;
    font-weight: 600;
    font-size: 15px;
    text-align: center;
  }
  .section-header {
    font-size: 18px;
    font-weight: 600;
    color: #1F4E79;
    border-bottom: 2px solid #2196F3;
    padding-bottom: 4px;
    margin: 16px 0 10px 0;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0;
    padding: 8px 20px;
  }
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING ───────────────────────────────────────────────
@st.cache_data(show_spinner="Loading prediction data...")
def load_predictions():
    for path in PRED_PATHS:
        if not os.path.exists(path):
            continue
        needed = ["latitude","longitude","NO2_pred_ugm3","NO2_ground_ugm3",
                  "date","season","month","station_name","sat_lat","sat_lon"]
        cols   = pd.read_csv(path, nrows=0).columns.tolist()
        use    = [c for c in needed if c in cols]
        df     = pd.read_csv(path, usecols=use)

        # Standardise column names
        if "sat_lat" in df.columns:
            df = df.rename(columns={"sat_lat":"latitude","sat_lon":"longitude"})
        if "NO2_ground_ugm3" in df.columns and "NO2_pred_ugm3" not in df.columns:
            df["NO2_pred_ugm3"] = df["NO2_ground_ugm3"]
        if "season" not in df.columns and "month" in df.columns:
            df["season"] = df["month"].map(
                {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:2,10:3,11:3})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["latitude","longitude","NO2_pred_ugm3"])
        df = df[df["NO2_pred_ugm3"].between(0, 500)]
        return df
    return None


@st.cache_data(show_spinner="Loading station data...")
def load_training():
    for path in TRAIN_PATHS:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        if "sat_lat" in df.columns:
            df = df.rename(columns={"sat_lat":"latitude","sat_lon":"longitude"})
        if "NO2_ground_ugm3" not in df.columns and "NO2_pred_ugm3" in df.columns:
            df["NO2_ground_ugm3"] = df["NO2_pred_ugm3"]
        if "season" not in df.columns and "month" in df.columns:
            df["season"] = df["month"].map(
                {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:2,10:3,11:3})
        return df
    return None


# ── HELPER FUNCTIONS ───────────────────────────────────────────
def get_aqi(no2):
    for lo, hi, color, label, msg in AQI_LEVELS:
        if no2 < hi:
            return color, label, msg
    return "#7E0023", "Severe", "Emergency conditions."


def metric_card(label, value, sub="", color="#2196F3"):
    return f"""
    <div class='metric-card' style='border-left-color:{color}'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value'>{value}</div>
      <div class='metric-sub'>{sub}</div>
    </div>"""


def make_scatter_map(lats, lons, vals, title, vmax=150, size=5):
    fig = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons,
        mode="markers",
        marker=dict(
            size=size,
            color=vals,
            colorscale=NO2_COLORSCALE,
            cmin=0, cmax=vmax,
            opacity=0.88,
            colorbar=dict(
                title="NO₂<br>(µg/m³)",
                thickness=12,
                len=0.7,
                tickvals=[0, 25, 80, min(vmax, 150)],
                ticktext=["0", "25 WHO", "80 NAAQS", f"{min(vmax,150):.0f}"],
                tickfont=dict(size=10),
            ),
        ),
        hovertemplate="<b>NO₂: %{marker.color:.1f} µg/m³</b><br>"
                      "Lat: %{lat:.4f} | Lon: %{lon:.4f}<extra></extra>",
    ))

    # Ground station markers
    for name, (lat, lon) in STATIONS.items():
        short = name.split(",")[0]
        fig.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon], mode="markers+text",
            marker=dict(size=12, color="navy"),
            text=[short], textposition="top right",
            textfont=dict(size=8, color="navy"),
            hovertemplate=f"<b>{short}</b><extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=14)),
        mapbox=dict(style="open-street-map",
                    center=dict(lat=18.52, lon=73.86), zoom=11),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════

def main():
    # ── LOAD DATA ──────────────────────────────────────────────
    df_pred  = load_predictions()
    df_train = load_training()

    if df_pred is None:
        st.error("No prediction data found. Run step2_train_and_predict.py first.")
        st.info("Expected file: data/output/all_predictions_500m.csv")
        st.stop()

    # ── SIDEBAR ────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/"
                 "Smiley.svg/40px-Smiley.svg.png",
                 width=30)  # placeholder — replace with your logo
        st.title("Pune NO₂ Monitor")
        st.caption("Sentinel-5P Downscaling | 500m Resolution")
        st.divider()

        # Date picker
        all_dates = sorted(df_pred["date"].unique())
        st.markdown("**Select Date**")
        selected_date = st.selectbox(
            "Date", all_dates, index=len(all_dates)-1,
            label_visibility="collapsed"
        )

        # Map style
        st.markdown("**Map Style**")
        map_style = st.selectbox(
            "Map", ["open-street-map","carto-positron",
                    "carto-darkmatter","stamen-terrain"],
            label_visibility="collapsed"
        )

        st.divider()

        # City-wide stats for selected date
        day_df = df_pred[df_pred["date"] == selected_date]
        if len(day_df) > 0:
            m_no2    = day_df["NO2_pred_ugm3"].mean()
            mx_no2   = day_df["NO2_pred_ugm3"].max()
            p_who    = (day_df["NO2_pred_ugm3"] > WHO_LIMIT).mean()   * 100
            p_naaqs  = (day_df["NO2_pred_ugm3"] > NAAQS_LIMIT).mean() * 100
            aqi_c, aqi_l, _ = get_aqi(m_no2)

            st.markdown("**City Statistics**")
            st.markdown(f"<div class='aqi-banner' style='background:{aqi_c};"
                        f"color:{'#000' if aqi_c in ['#00E400','#FFFF00'] else '#fff'}'>"
                        f"AQI: {aqi_l}</div>", unsafe_allow_html=True)

            st.markdown(metric_card("Mean NO₂", f"{m_no2:.1f} µg/m³",
                                    f"across {len(day_df):,} pixels"), unsafe_allow_html=True)
            st.markdown(metric_card("Max NO₂", f"{mx_no2:.1f} µg/m³",
                                    "highest pixel in Pune", "#E53935"), unsafe_allow_html=True)
            st.markdown(metric_card("Above WHO (25)", f"{p_who:.1f}%",
                                    "of pixels exceed guideline", "#FF9800"), unsafe_allow_html=True)
            st.markdown(metric_card("Above NAAQS (80)", f"{p_naaqs:.1f}%",
                                    "of pixels exceed standard", "#E53935"), unsafe_allow_html=True)

        st.divider()

        # Reference limits toggle
        show_limits = st.checkbox("Show WHO / NAAQS lines on charts", value=True)

        st.divider()
        st.caption("Data: Sentinel-5P TROPOMI + CPCB ground stations")
        st.caption("Model: Random Forest | R²=0.652 | RMSE=25.86 µg/m³")

    # ── MAIN AREA HEADER ───────────────────────────────────────
    col_title, col_export = st.columns([5, 1])
    with col_title:
        st.title(f"🏭 Pune NO₂ Air Quality — {selected_date}")
    with col_export:
        if len(day_df) > 0:
            csv_data = day_df[["latitude","longitude","NO2_pred_ugm3"]]\
                           .round(4).to_csv(index=False)
            st.download_button(
                label="⬇ Export CSV",
                data=csv_data,
                file_name=f"pune_no2_{selected_date}.csv",
                mime="text/csv",
            )

    # ── AQI HEALTH ADVISORY BANNER ─────────────────────────────
    if len(day_df) > 0:
        aqi_color, aqi_label, aqi_msg = get_aqi(m_no2)
        text_color = "#000" if aqi_color in ["#00E400","#FFFF00"] else "#fff"
        st.markdown(
            f"<div style='background:{aqi_color};color:{text_color};"
            f"border-radius:8px;padding:10px 20px;margin-bottom:12px;"
            f"font-size:15px;font-weight:600'>"
            f"🌬 Air Quality: <b>{aqi_label}</b> — {aqi_msg}"
            f"</div>",
            unsafe_allow_html=True
        )

    # ── TABS ───────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📍 Daily Map",
        "📅 Annual Map",
        "🌦 Seasonal Maps",
        "📊 Station Analysis",
        "📈 Time Series",
        "⚖️ Season Comparison",
    ])


    # ═══════════════════════════════════════════════
    #  TAB 1 — DAILY MAP
    # ═══════════════════════════════════════════════
    with tab1:
        st.markdown(f"<div class='section-header'>Daily NO₂ Map — {selected_date}</div>",
                    unsafe_allow_html=True)

        if len(day_df) == 0:
            st.warning(f"No satellite data for {selected_date} (likely cloudy day).")
        else:
            col_map, col_hist = st.columns([3, 1])

            with col_map:
                vmax_day = min(float(day_df["NO2_pred_ugm3"].quantile(0.97)), 200)
                fig_day  = make_scatter_map(
                    day_df["latitude"], day_df["longitude"],
                    day_df["NO2_pred_ugm3"],
                    f"Surface NO₂ — {selected_date}",
                    vmax=vmax_day
                )
                fig_day.update_layout(mapbox_style=map_style)
                st.plotly_chart(fig_day, use_container_width=True)

            with col_hist:
                st.markdown("**NO₂ Distribution**")
                # Distribution histogram
                fig_hist = go.Figure(go.Histogram(
                    x=day_df["NO2_pred_ugm3"],
                    nbinsx=30,
                    marker_color="#2196F3",
                    opacity=0.8,
                    name="Pixels",
                ))
                if show_limits:
                    fig_hist.add_vline(x=WHO_LIMIT,   line_color="orange",
                                       line_dash="dash", annotation_text="WHO")
                    fig_hist.add_vline(x=NAAQS_LIMIT, line_color="red",
                                       line_dash="dash", annotation_text="NAAQS")
                fig_hist.update_layout(
                    height=220, margin=dict(l=0,r=0,t=10,b=0),
                    xaxis_title="NO₂ (µg/m³)", yaxis_title="Count",
                    showlegend=False,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # AQI breakdown pie chart
                st.markdown("**AQI Breakdown**")
                aqi_counts = {}
                for lo, hi, color, label, _ in AQI_LEVELS:
                    count = ((day_df["NO2_pred_ugm3"] >= lo) &
                             (day_df["NO2_pred_ugm3"] < hi)).sum()
                    if count > 0:
                        aqi_counts[label] = (count, color)

                if aqi_counts:
                    fig_pie = go.Figure(go.Pie(
                        labels=list(aqi_counts.keys()),
                        values=[v[0] for v in aqi_counts.values()],
                        marker_colors=[v[1] for v in aqi_counts.values()],
                        hole=0.4,
                        textinfo="percent",
                        textfont_size=10,
                    ))
                    fig_pie.update_layout(
                        height=220, margin=dict(l=0,r=0,t=10,b=10),
                        showlegend=True,
                        legend=dict(font=dict(size=9), x=0, y=0),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            # Station vs predicted on this day
            if df_train is not None:
                day_train = df_train[df_train["date"] == selected_date]
                if len(day_train) > 0:
                    st.markdown("**Ground Station Readings vs Model Prediction**")
                    short_names = day_train["station_name"].str.split(",").str[0]\
                                       .str.split("_").str[0]

                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        name="Ground measured",
                        x=short_names,
                        y=day_train["NO2_ground_ugm3"].round(1),
                        marker_color="#2196F3", opacity=0.85,
                    ))
                    if "NO2_pred_ugm3" in day_train.columns:
                        fig_bar.add_trace(go.Bar(
                            name="Model predicted",
                            x=short_names,
                            y=day_train["NO2_pred_ugm3"].round(1),
                            marker_color="#FF5722", opacity=0.85,
                        ))
                    if show_limits:
                        fig_bar.add_hline(y=WHO_LIMIT,   line_dash="dash",
                                          line_color="orange",
                                          annotation_text="WHO 25")
                        fig_bar.add_hline(y=NAAQS_LIMIT, line_dash="dash",
                                          line_color="red",
                                          annotation_text="NAAQS 80")
                    fig_bar.update_layout(
                        barmode="group", height=300,
                        margin=dict(l=0,r=0,t=10,b=0),
                        yaxis_title="NO₂ (µg/m³)",
                        legend=dict(orientation="h", y=1.02),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)


    # ═══════════════════════════════════════════════
    #  TAB 2 — ANNUAL MAP
    # ═══════════════════════════════════════════════
    with tab2:
        st.markdown("<div class='section-header'>Annual Mean NO₂ — All Available Dates</div>",
                    unsafe_allow_html=True)

        annual = df_pred.groupby(["latitude","longitude"],
                                  as_index=False)["NO2_pred_ugm3"].mean()
        vmax_a = min(float(annual["NO2_pred_ugm3"].quantile(0.97)), 200)

        col_am, col_as = st.columns([3, 1])
        with col_am:
            fig_annual = make_scatter_map(
                annual["latitude"], annual["longitude"],
                annual["NO2_pred_ugm3"],
                f"Annual Mean Surface NO₂ | {len(annual):,} pixels | "
                f"{df_pred['date'].nunique()} satellite days",
                vmax=vmax_a
            )
            fig_annual.update_layout(mapbox_style=map_style)
            st.plotly_chart(fig_annual, use_container_width=True)

        with col_as:
            mn = annual["NO2_pred_ugm3"].mean()
            mx = annual["NO2_pred_ugm3"].max()
            pw = (annual["NO2_pred_ugm3"] > WHO_LIMIT).mean()   * 100
            pn = (annual["NO2_pred_ugm3"] > NAAQS_LIMIT).mean() * 100

            st.metric("Mean NO₂",        f"{mn:.1f} µg/m³")
            st.metric("Peak NO₂",        f"{mx:.1f} µg/m³")
            st.metric("Above WHO",       f"{pw:.1f}%")
            st.metric("Above NAAQS",     f"{pn:.1f}%")
            st.metric("Pixels mapped",   f"{len(annual):,}")
            st.metric("Days covered",    f"{df_pred['date'].nunique()}")

            # Monthly trend
            if "month" in df_pred.columns:
                st.markdown("**Monthly mean**")
                monthly = df_pred.groupby("month")["NO2_pred_ugm3"].mean().reset_index()
                month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"]
                monthly["month_name"] = monthly["month"].apply(
                    lambda m: month_labels[m-1])
                fig_m = go.Figure(go.Bar(
                    x=monthly["month_name"],
                    y=monthly["NO2_pred_ugm3"].round(1),
                    marker_color="#2196F3", opacity=0.8,
                ))
                if show_limits:
                    fig_m.add_hline(y=WHO_LIMIT,   line_dash="dash",
                                    line_color="orange")
                    fig_m.add_hline(y=NAAQS_LIMIT, line_dash="dash",
                                    line_color="red")
                fig_m.update_layout(
                    height=220, margin=dict(l=0,r=0,t=10,b=0),
                    yaxis_title="µg/m³", showlegend=False,
                )
                st.plotly_chart(fig_m, use_container_width=True)

        # Exceedance map
        st.markdown("---")
        st.markdown("<div class='section-header'>NAAQS Exceedance Map</div>",
                    unsafe_allow_html=True)
        exc = df_pred.groupby(["latitude","longitude"],as_index=False).apply(
            lambda g: pd.Series({
                "pct_who":   round((g["NO2_pred_ugm3"]>WHO_LIMIT).mean()*100,1),
                "pct_naaqs": round((g["NO2_pred_ugm3"]>NAAQS_LIMIT).mean()*100,1),
            })
        ).reset_index(drop=True)

        col_e1, col_e2 = st.columns(2)
        for col_e, col_name, title in [
            (col_e1, "pct_who",   "WHO (25 µg/m³) — % days exceeded"),
            (col_e2, "pct_naaqs", "NAAQS (80 µg/m³) — % days exceeded"),
        ]:
            with col_e:
                fig_e = go.Figure(go.Scattermapbox(
                    lat=exc["latitude"], lon=exc["longitude"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=np.clip(exc[col_name],0,100),
                        colorscale="YlOrRd",
                        cmin=0, cmax=100,
                        opacity=0.88,
                        colorbar=dict(title="% days",thickness=10,len=0.6),
                    ),
                    hovertemplate=f"<b>{col_name.replace('_',' ').title()}: "
                                  "%{marker.color:.1f}%</b><extra></extra>",
                ))
                fig_e.update_layout(
                    title=dict(text=title, x=0.01, font=dict(size=12)),
                    mapbox=dict(style=map_style,
                                center=dict(lat=18.52,lon=73.86), zoom=10),
                    margin=dict(l=0,r=0,t=35,b=0), height=380,
                )
                st.plotly_chart(fig_e, use_container_width=True)


    # ═══════════════════════════════════════════════
    #  TAB 3 — SEASONAL MAPS
    # ═══════════════════════════════════════════════
    with tab3:
        st.markdown("<div class='section-header'>Seasonal NO₂ Patterns</div>",
                    unsafe_allow_html=True)

        if "season" not in df_pred.columns:
            st.warning("Season column not found in predictions.")
        else:
            seasons_data = {}
            for s_code, s_name in SEASON_NAMES.items():
                sub = df_pred[df_pred["season"] == s_code]
                if len(sub) > 0:
                    smean = sub.groupby(["latitude","longitude"],
                                         as_index=False)["NO2_pred_ugm3"].mean()
                    seasons_data[s_name] = (smean, sub["date"].nunique(),
                                            smean["NO2_pred_ugm3"].mean())

            # Season summary bar
            fig_summary = go.Figure()
            for s_name, (_, n_days, mean_val) in seasons_data.items():
                fig_summary.add_trace(go.Bar(
                    name=s_name, x=[s_name], y=[round(mean_val,1)],
                    marker_color=SEASON_COLORS.get(s_name,"#888"),
                    text=[f"{mean_val:.1f}"], textposition="outside",
                ))
            if show_limits:
                fig_summary.add_hline(y=WHO_LIMIT,   line_dash="dash",
                                      line_color="orange",
                                      annotation_text="WHO 25")
                fig_summary.add_hline(y=NAAQS_LIMIT, line_dash="dash",
                                      line_color="red",
                                      annotation_text="NAAQS 80")
            fig_summary.update_layout(
                height=200, showlegend=False,
                yaxis_title="Mean NO₂ (µg/m³)",
                margin=dict(l=0,r=0,t=10,b=0),
                title="Mean NO₂ by Season — all Pune pixels",
            )
            st.plotly_chart(fig_summary, use_container_width=True)

            # Season selector
            sel_season = st.selectbox(
                "Select season to view",
                list(seasons_data.keys())
            )

            if sel_season in seasons_data:
                smean, n_days, mean_val = seasons_data[sel_season]
                vmax_s = min(float(smean["NO2_pred_ugm3"].quantile(0.97)), 200)

                col_sm, col_ss = st.columns([3, 1])
                with col_sm:
                    fig_s = make_scatter_map(
                        smean["latitude"], smean["longitude"],
                        smean["NO2_pred_ugm3"],
                        f"{sel_season} mean NO₂ | {n_days} satellite days | "
                        f"mean={mean_val:.1f} µg/m³",
                        vmax=vmax_s
                    )
                    fig_s.update_layout(mapbox_style=map_style)
                    st.plotly_chart(fig_s, use_container_width=True)

                with col_ss:
                    st.metric("Season", sel_season)
                    st.metric("Mean NO₂",    f"{mean_val:.1f} µg/m³")
                    st.metric("Max NO₂",     f"{smean['NO2_pred_ugm3'].max():.1f} µg/m³")
                    st.metric("Days covered", n_days)
                    st.metric("% above WHO",
                              f"{(smean['NO2_pred_ugm3']>WHO_LIMIT).mean()*100:.1f}%")
                    st.metric("% above NAAQS",
                              f"{(smean['NO2_pred_ugm3']>NAAQS_LIMIT).mean()*100:.1f}%")

                    # Histogram for season
                    fig_sh = go.Figure(go.Histogram(
                        x=smean["NO2_pred_ugm3"], nbinsx=25,
                        marker_color=SEASON_COLORS.get(sel_season,"#2196F3"),
                        opacity=0.8,
                    ))
                    if show_limits:
                        fig_sh.add_vline(x=WHO_LIMIT,   line_dash="dash",
                                         line_color="orange")
                        fig_sh.add_vline(x=NAAQS_LIMIT, line_dash="dash",
                                         line_color="red")
                    fig_sh.update_layout(
                        height=200, margin=dict(l=0,r=0,t=10,b=0),
                        xaxis_title="NO₂ (µg/m³)", showlegend=False,
                    )
                    st.plotly_chart(fig_sh, use_container_width=True)


    # ═══════════════════════════════════════════════
    #  TAB 4 — STATION ANALYSIS
    # ═══════════════════════════════════════════════
    with tab4:
        st.markdown("<div class='section-header'>Ground Station Analysis</div>",
                    unsafe_allow_html=True)

        if df_train is None:
            st.warning("Training data not found.")
        else:
            # Station selector
            all_stations = sorted(df_train["station_name"].unique()) \
                           if "station_name" in df_train.columns else []

            if all_stations:
                sel_station = st.selectbox("Select station", all_stations)
                st_df = df_train[df_train["station_name"] == sel_station].copy()
                st_df = st_df.sort_values("date")

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Mean measured", f"{st_df['NO2_ground_ugm3'].mean():.1f} µg/m³")
                col_s2.metric("Max measured",  f"{st_df['NO2_ground_ugm3'].max():.1f} µg/m³")
                col_s3.metric("Data days",     len(st_df))
                col_s4.metric("Above NAAQS",
                              f"{(st_df['NO2_ground_ugm3']>NAAQS_LIMIT).mean()*100:.1f}%")

                # Predicted vs Observed scatter
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("**Predicted vs Observed**")
                    if "NO2_pred_ugm3" in st_df.columns:
                        fig_sc = go.Figure()
                        fig_sc.add_trace(go.Scatter(
                            x=st_df["NO2_ground_ugm3"],
                            y=st_df["NO2_pred_ugm3"],
                            mode="markers",
                            marker=dict(size=6, color="#2196F3",
                                        opacity=0.7),
                            name="Pixel predictions",
                        ))
                        lim = max(st_df["NO2_ground_ugm3"].max(),
                                  st_df["NO2_pred_ugm3"].max()) * 1.05
                        fig_sc.add_trace(go.Scatter(
                            x=[0,lim], y=[0,lim],
                            mode="lines",
                            line=dict(color="red", dash="dash"),
                            name="Perfect 1:1",
                        ))
                        fig_sc.update_layout(
                            height=320,
                            xaxis_title="Measured NO₂ (µg/m³)",
                            yaxis_title="Predicted NO₂ (µg/m³)",
                            margin=dict(l=0,r=0,t=10,b=0),
                        )
                        st.plotly_chart(fig_sc, use_container_width=True)

                with col_t2:
                    st.markdown("**All Stations — Mean NO₂ Comparison**")
                    station_means = df_train.groupby("station_name")[
                        "NO2_ground_ugm3"].mean().sort_values(ascending=True)
                    short_names   = [s.split(",")[0].split("_")[0]
                                     for s in station_means.index]
                    colors_bar    = ["#E53935" if v > NAAQS_LIMIT
                                     else "#FF9800" if v > WHO_LIMIT
                                     else "#4CAF50"
                                     for v in station_means.values]

                    fig_all = go.Figure(go.Bar(
                        x=station_means.values.round(1),
                        y=short_names,
                        orientation="h",
                        marker_color=colors_bar,
                        text=station_means.values.round(1),
                        textposition="outside",
                    ))
                    if show_limits:
                        fig_all.add_vline(x=WHO_LIMIT,   line_dash="dash",
                                          line_color="orange")
                        fig_all.add_vline(x=NAAQS_LIMIT, line_dash="dash",
                                          line_color="red")
                    fig_all.update_layout(
                        height=320, margin=dict(l=0,r=0,t=10,b=0),
                        xaxis_title="Mean NO₂ (µg/m³)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_all, use_container_width=True)


    # ═══════════════════════════════════════════════
    #  TAB 5 — TIME SERIES
    # ═══════════════════════════════════════════════
    with tab5:
        st.markdown("<div class='section-header'>NO₂ Time Series Analysis</div>",
                    unsafe_allow_html=True)

        # City-wide time series from predictions
        if "date" in df_pred.columns:
            ts = df_pred.groupby("date")["NO2_pred_ugm3"]\
                         .agg(["mean","max","min"]).reset_index()
            ts = ts.sort_values("date")

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=ts["date"], y=ts["min"],
                fill=None, mode="lines",
                line=dict(color="rgba(33,150,243,0)"),
                showlegend=False,
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts["date"], y=ts["max"],
                fill="tonexty",
                fillcolor="rgba(33,150,243,0.12)",
                mode="lines",
                line=dict(color="rgba(33,150,243,0)"),
                name="Min–Max range",
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts["date"], y=ts["mean"].round(2),
                mode="lines+markers",
                line=dict(color="#1565C0", width=2),
                marker=dict(size=4),
                name="Daily mean",
            ))
            if show_limits:
                fig_ts.add_hline(y=WHO_LIMIT,   line_dash="dash",
                                 line_color="orange",
                                 annotation_text="WHO 25")
                fig_ts.add_hline(y=NAAQS_LIMIT, line_dash="dash",
                                 line_color="red",
                                 annotation_text="NAAQS 80")

            # Mark selected date
            sel_val = ts[ts["date"]==selected_date]["mean"]
            if len(sel_val) > 0:
                fig_ts.add_trace(go.Scatter(
                    x=[selected_date],
                    y=[round(float(sel_val.values[0]),1)],
                    mode="markers",
                    marker=dict(size=12, color="red", symbol="star"),
                    name=f"Selected: {selected_date}",
                ))

            fig_ts.update_layout(
                title="City-wide daily mean NO₂ over time",
                height=380, margin=dict(l=0,r=0,t=40,b=0),
                xaxis_title="Date",
                yaxis_title="NO₂ (µg/m³)",
                legend=dict(orientation="h", y=1.05),
                hovermode="x unified",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        # Station time series
        if df_train is not None and "station_name" in df_train.columns:
            st.markdown("---")
            st.markdown("**Per-Station Time Series**")

            stations_avail = sorted(df_train["station_name"].unique())
            sel_stations   = st.multiselect(
                "Select stations to compare",
                stations_avail,
                default=stations_avail[:3],
            )

            if sel_stations:
                fig_sts = go.Figure()
                for stn in sel_stations:
                    sdf = df_train[df_train["station_name"]==stn]\
                              .sort_values("date")
                    short = stn.split(",")[0].split("_")[0]
                    fig_sts.add_trace(go.Scatter(
                        x=sdf["date"],
                        y=sdf["NO2_ground_ugm3"].round(1),
                        mode="lines+markers",
                        marker=dict(size=4),
                        name=short,
                    ))
                if show_limits:
                    fig_sts.add_hline(y=WHO_LIMIT,   line_dash="dash",
                                      line_color="orange")
                    fig_sts.add_hline(y=NAAQS_LIMIT, line_dash="dash",
                                      line_color="red")
                fig_sts.update_layout(
                    height=360, margin=dict(l=0,r=0,t=10,b=0),
                    yaxis_title="NO₂ (µg/m³)",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.02),
                )
                st.plotly_chart(fig_sts, use_container_width=True)


    # ═══════════════════════════════════════════════
    #  TAB 6 — SEASON COMPARISON (side by side)
    # ═══════════════════════════════════════════════
    with tab6:
        st.markdown("<div class='section-header'>Season-by-Season Comparison</div>",
                    unsafe_allow_html=True)

        if "season" not in df_pred.columns:
            st.warning("Season column not found.")
        else:
            col_left, col_right = st.columns(2)
            with col_left:
                s_left  = st.selectbox("Left map season",
                                       list(SEASON_NAMES.values()),
                                       index=0, key="sl")
            with col_right:
                s_right = st.selectbox("Right map season",
                                       list(SEASON_NAMES.values()),
                                       index=2, key="sr")

            # Build both maps
            s_code_map = {v:k for k,v in SEASON_NAMES.items()}

            def get_season_fig(season_name, map_key):
                s_code = s_code_map[season_name]
                sub    = df_pred[df_pred["season"]==s_code]
                if len(sub) == 0:
                    return None, 0, 0
                smean = sub.groupby(["latitude","longitude"],
                                     as_index=False)["NO2_pred_ugm3"].mean()
                mean_v = smean["NO2_pred_ugm3"].mean()
                vmax_v = min(float(smean["NO2_pred_ugm3"].quantile(0.97)),200)
                fig = go.Figure(go.Scattermapbox(
                    lat=smean["latitude"], lon=smean["longitude"],
                    mode="markers",
                    marker=dict(size=4.5,
                                color=smean["NO2_pred_ugm3"],
                                colorscale=NO2_COLORSCALE,
                                cmin=0, cmax=vmax_v,
                                opacity=0.88,
                                colorbar=dict(title="NO₂",thickness=10,len=0.6)),
                    hovertemplate="NO₂: %{marker.color:.1f} µg/m³<extra></extra>",
                ))
                fig.update_layout(
                    title=dict(
                        text=f"<b>{season_name}</b> | mean={mean_v:.1f} µg/m³ | "
                             f"{sub['date'].nunique()} days",
                        x=0.01, font=dict(size=12)),
                    mapbox=dict(style=map_style,
                                center=dict(lat=18.52,lon=73.86), zoom=10),
                    margin=dict(l=0,r=0,t=40,b=0), height=420,
                )
                return fig, mean_v, vmax_v

            fig_l, mean_l, _ = get_season_fig(s_left,  "left")
            fig_r, mean_r, _ = get_season_fig(s_right, "right")

            col_left, col_right = st.columns(2)
            if fig_l:
                with col_left:
                    st.plotly_chart(fig_l, use_container_width=True)
            if fig_r:
                with col_right:
                    st.plotly_chart(fig_r, use_container_width=True)

            # Difference metric
            if mean_l > 0 and mean_r > 0:
                diff = mean_l - mean_r
                pct  = abs(diff) / max(mean_l, mean_r) * 100
                st.info(
                    f"**{s_left}** mean: **{mean_l:.1f} µg/m³**  |  "
                    f"**{s_right}** mean: **{mean_r:.1f} µg/m³**  |  "
                    f"Difference: **{abs(diff):.1f} µg/m³ ({pct:.1f}%)**  "
                    f"({'Winter has more' if s_left=='Winter' and diff>0 else 'Monsoon cleanest' if s_right=='Monsoon' and diff>0 else 'Higher in ' + (s_left if diff>0 else s_right)})"
                )


# ── RUN ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
