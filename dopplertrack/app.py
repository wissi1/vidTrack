"""Streamlit Web UI for dopplertrack:
- Part 1: Audio / Spectrogram Doppler Curve Analysis & Event Fitting
- Part 2: Flight Matching & Geolocation Circle Search
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dopplertrack.analyzer import (
    circle_to_bbox,
    get_circle_perimeter,
    load_events_json,
    process_audio_file,
    run_adsb_search,
    run_detection,
    run_window_fit,
    save_events_json,
)
from dopplertrack.audio import Spectrogram
from dopplertrack.search import DayIndex
from dopplertrack.tracks import EventFit
from dopplertrack.viz import (
    format_seconds,
    plot_geolocation_map,
    plot_spectrogram_plotly,
)

st.set_page_config(
    page_title="DopplerTrack — Acoustic Aircraft Geolocation",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ DopplerTrack — Acoustic Aircraft Geolocation")
st.caption(
    "Detect inaudible propeller Doppler sweeps from video/audio and geolocate recordings "
    "by matching acoustic curves against historical ADS-B flight tracks."
)

# ----------------- SESSION STATE -----------------
if "spectrogram" not in st.session_state:
    st.session_state.spectrogram = None
if "events" not in st.session_state:
    st.session_state.events = []
if "spec_source" not in st.session_state:
    st.session_state.spec_source = ""
if "media_path" not in st.session_state:
    st.session_state.media_path = None
if "day_index" not in st.session_state:
    st.session_state.day_index = None
if "day_source" not in st.session_state:
    st.session_state.day_source = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "live_df" not in st.session_state:
    st.session_state.live_df = None

# Auto-load detected.json if events are empty
if not st.session_state.events:
    for candidate_json in ("detected.json", "ev1.json"):
        if Path(candidate_json).exists():
            try:
                st.session_state.events = load_events_json(candidate_json)
                break
            except Exception:
                pass

# Auto-load rec.npz if spectrogram is empty
if st.session_state.spectrogram is None and Path("rec.npz").exists():
    try:
        st.session_state.spectrogram = Spectrogram.load("rec.npz")
        st.session_state.spec_source = "rec.npz (auto-loaded)"
    except Exception:
        pass

# Auto-load day.npz if day_index is empty
if st.session_state.day_index is None and Path("day.npz").exists():
    try:
        st.session_state.day_index = DayIndex.load("day.npz")
        st.session_state.day_source = "day.npz (auto-loaded)"
    except Exception:
        pass


# ----------------- MAIN NAVIGATION TABS -----------------
tab_part1, tab_part2 = st.tabs(
    [
        "🎵 Part 1: Audio & Doppler Curve Analyzer",
        "🗺️ Part 2: Flight Matching & Circle Search",
    ]
)


# ========================================================================================
# PART 1: AUDIO & SPECTROGRAM ANALYZER
# ========================================================================================
with tab_part1:
    st.header("1️⃣ Audio Spectrogram & Doppler Detection")

    # Source Selection
    sub_tab_existing, sub_tab_new = st.tabs(["📂 Load Existing Spectrogram (.npz)", "🎬 Process Audio / Video File"])

    with sub_tab_existing:
        col_e1, col_e2 = st.columns([3, 1])
        with col_e1:
            local_npz_files = [str(p) for p in Path(".").glob("*.npz") if p.name != "day.npz" and p.name != "cands.npz"]
            if "rec.npz" not in local_npz_files and Path("rec.npz").exists():
                local_npz_files.insert(0, "rec.npz")

            selected_npz = st.selectbox(
                "Select existing spectrogram file:",
                options=local_npz_files if local_npz_files else ["rec.npz"],
                key="part1_sel_npz",
            )
        with col_e2:
            st.write("")
            st.write("")
            if st.button("Load Spectrogram", key="btn_load_npz", use_container_width=True):
                if selected_npz and os.path.exists(selected_npz):
                    try:
                        with st.spinner("Loading spectrogram..."):
                            st.session_state.spectrogram = Spectrogram.load(selected_npz)
                            st.session_state.spec_source = selected_npz
                            st.success(f"Loaded: `{selected_npz}`")
                    except Exception as e:
                        st.error(f"Error loading {selected_npz}: {e}")

    with sub_tab_new:
        col_u1, col_u2, col_u3 = st.columns([2, 1, 1])
        with col_u1:
            uploaded_file = st.file_uploader(
                "Upload audio/video (MP4, MKV, WAV, M4A, OPUS):",
                type=["mp4", "mkv", "mov", "avi", "wav", "m4a", "opus", "flac"],
                key="part1_upload",
            )
            media_path_input = st.text_input("Or enter local file path:", "", key="part1_media_path")
        with col_u2:
            hop_s = st.slider("Hop size (s)", min_value=0.05, max_value=0.50, value=0.15, step=0.05, help="Smaller hop (0.10s) gives sharper time resolution for short clips.")
            fmin = st.number_input("Fmin (Hz)", min_value=20.0, max_value=80.0, value=40.0, step=5.0)
            fmax = st.number_input("Fmax (Hz)", min_value=100.0, max_value=600.0, value=450.0, step=25.0)
        with col_u3:
            st.write("")
            st.write("")
            process_btn = st.button("⚡ Extract & Whiten", key="btn_process_media", use_container_width=True)

        if process_btn:
            source_path = None
            if uploaded_file is not None:
                suffix = Path(uploaded_file.name).suffix
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tfile.write(uploaded_file.read())
                tfile.close()
                source_path = tfile.name
            elif media_path_input.strip() and os.path.exists(media_path_input.strip()):
                source_path = media_path_input.strip()

            if source_path:
                with st.spinner("Decoding audio via ffmpeg & computing whitened STFT..."):
                    try:
                        out_npz = "rec.npz"
                        sp = process_audio_file(source_path, out_npz_path=out_npz, hop_s=hop_s, fmin=fmin, fmax=fmax)
                        st.session_state.spectrogram = sp
                        st.session_state.spec_source = f"{source_path} -> {out_npz}"
                        st.success(f"Spectrogram saved to `{out_npz}`!")
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
            else:
                st.error("Please upload a file or enter a valid local path.")

    st.markdown("---")

    # Spectrogram View & Event Controls
    sp: Spectrogram | None = st.session_state.spectrogram
    if sp is not None:
        duration_s = float(sp.t[-1] - sp.t[0])
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Duration", format_seconds(duration_s), f"{duration_s:.1f} s")
        col_m2.metric("Band", f"{sp.f[0]:.1f} – {sp.f[-1]:.1f} Hz", f"{len(sp.f)} bins")
        col_m3.metric("Time Step (dt)", f"{sp.dt:.2f} s", f"{len(sp.t)} frames")
        col_m4.metric("Freq Step (df)", f"{sp.df:.2f} Hz")
        col_m5.metric("Active Events", f"{len(st.session_state.events)}")

        st.subheader("🔍 Interactive Spectrogram Viewport")
        col_vp1, col_vp2, col_vp3 = st.columns([2, 2, 1])
        with col_vp1:
            t_min_val, t_max_val = float(sp.t[0]), float(sp.t[-1])
            t_range = st.slider(
                "Time Viewport (seconds):",
                min_value=t_min_val,
                max_value=t_max_val,
                value=(t_min_val, min(t_min_val + 300.0, t_max_val)),
                step=1.0,
                key="part1_tslider",
            )
        with col_vp2:
            f_min_val, f_max_val = float(sp.f[0]), float(sp.f[-1])
            f_range = st.slider(
                "Frequency Viewport (Hz):",
                min_value=f_min_val,
                max_value=f_max_val,
                value=(f_min_val, f_max_val),
                step=5.0,
                key="part1_fslider",
            )
        with col_vp3:
            colorscale = st.selectbox("Colormap", ["Magma", "Viridis", "Inferno", "Turbo"], index=0, key="part1_cmap")
            harmonics_disp = st.slider("Harmonics to plot", 1, 3, 2, key="part1_hdisp")

        # Event selection for overlay
        active_events = []
        if st.session_state.events:
            event_labels = [
                f"#{i+1}: CPA {format_seconds(ev.t_cpa)} ({ev.t_cpa:.1f}s), f0={ev.f0:.1f}Hz, v={ev.v:.0f}m/s, d={ev.d:.0f}m"
                for i, ev in enumerate(st.session_state.events)
            ]
            selected_idxs = st.multiselect(
                "Overlay Curves on Spectrogram:",
                options=list(range(len(event_labels))),
                default=list(range(min(len(event_labels), 3))),
                format_func=lambda i: event_labels[i],
                key="part1_active_evs",
            )
            active_events = [st.session_state.events[i] for i in selected_idxs]

        # Interactive Plotly chart
        fig_spec = plot_spectrogram_plotly(
            sp,
            events=active_events,
            t_range=t_range,
            f_range=f_range,
            harmonics=harmonics_disp,
            colorscale=colorscale,
            title=f"Doppler Spectrogram — {st.session_state.spec_source}",
        )
        st.plotly_chart(fig_spec, use_container_width=True)

        # Detection & Fitting Tabs
        st.markdown("---")
        p1_det_tab, p1_fit_tab, p1_list_tab = st.tabs(
            ["🤖 Automatic Detection (`detect`)", "📐 Manual Window Fitting (`fit`)", "📋 Events List"]
        )

        with p1_det_tab:
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                band_lo = st.number_input("Search Band Low (Hz)", value=60.0, step=5.0, key="p1_blo")
                band_hi = st.number_input("Search Band High (Hz)", value=300.0, step=10.0, key="p1_bhi")
            with col_d2:
                top_k = st.slider("Top candidates", 1, 20, 10, key="p1_topk")
                harmonics_det = st.selectbox("Harmonics", [1, 2], index=0, key="p1_hdet")
            with col_d3:
                window_size = st.slider("Window (s)", 30, 300, min(120, max(30, int(duration_s))), step=15, key="p1_ws")
                step_size = st.slider("Step (s)", 15, 150, min(60, max(15, int(duration_s / 2))), step=15, key="p1_ss")
            with col_d4:
                max_burst = st.slider("Burst filter threshold", 0.0, 0.5, 0.20, step=0.02, key="p1_mb")
                run_detect_btn = st.button("🚀 Run Auto Detection", use_container_width=True, type="primary", key="btn_p1_det")

            if run_detect_btn:
                with st.spinner("Scanning spectrogram for Doppler curves..."):
                    try:
                        found_events = run_detection(
                            sp,
                            f_lo=band_lo,
                            f_hi=band_hi,
                            harmonics=harmonics_det,
                            top=top_k,
                            window_s=float(window_size),
                            step_s=float(step_size),
                            max_burst=max_burst,
                        )
                        st.session_state.events = found_events
                        save_events_json(found_events, "detected.json")
                        st.success(f"Found {len(found_events)} events! Saved to `detected.json`.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Detection error: {e}")

        with p1_fit_tab:
            st.write("Fit a specific curve by selecting a bounding box around a sweep:")
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                fit_t0 = st.number_input("Start Time t0 (s)", min_value=float(sp.t[0]), max_value=float(sp.t[-1]), value=float(t_range[0]), step=2.0, key="fit_t0")
                fit_t1 = st.number_input("End Time t1 (s)", min_value=float(sp.t[0]), max_value=float(sp.t[-1]), value=min(float(t_range[0]) + 80.0, float(sp.t[-1])), step=2.0, key="fit_t1")
            with col_f2:
                fit_flo = st.number_input("Freq Low (Hz)", min_value=10.0, max_value=450.0, value=90.0, step=5.0, key="fit_flo")
                fit_fhi = st.number_input("Freq High (Hz)", min_value=10.0, max_value=450.0, value=250.0, step=5.0, key="fit_fhi")
            with col_f3:
                fit_h = st.selectbox("Harmonics to fit", [1, 2, 3], index=0, key="fit_h_sel")
                run_fit_btn = st.button("🔍 Fit Region", use_container_width=True, type="primary", key="btn_p1_fit")

            if run_fit_btn:
                with st.spinner("Fitting ideal Doppler curve..."):
                    try:
                        new_ev = run_window_fit(sp, fit_t0, fit_t1, fit_flo, fit_fhi, harmonics=fit_h)
                        st.session_state.events.append(new_ev)
                        save_events_json(st.session_state.events, "detected.json")
                        st.success(f"Fitted Event: CPA {new_ev.t_cpa:.1f}s | f0={new_ev.f0:.1f}Hz | v={new_ev.v*3.6:.0f}km/h | d={new_ev.d:.0f}m")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fit error: {e}")

        with p1_list_tab:
            if st.session_state.events:
                t_rows = []
                for idx_ev, ev in enumerate(st.session_state.events, 1):
                    t_rows.append({
                        "Event": f"#{idx_ev}",
                        "CPA Time": format_seconds(ev.t_cpa),
                        "CPA (s)": round(ev.t_cpa, 2),
                        "f0 (Hz)": round(ev.f0, 2),
                        "Speed (km/h)": round(ev.v * 3.6, 1),
                        "Speed (kts)": round(ev.v * 1.94384, 1),
                        "Slant Range (m)": round(ev.d, 0),
                        "Score (dB)": round(ev.score, 2),
                    })
                st.dataframe(pd.DataFrame(t_rows), use_container_width=True)
            else:
                st.info("No events loaded yet.")
    else:
        st.info("👆 Please select or process a spectrogram above to begin.")


# ========================================================================================
# PART 2: FLIGHT MATCHING & LOCATION CIRCLE SEARCH
# ========================================================================================
with tab_part2:
    st.header("2️⃣ Flight Matching & Circle Geolocation Search")
    st.markdown(
        "Match your detected Doppler events against real historical flight tracks in a specific "
        "**date, time window, and geographic circle** to geolocate the recording."
    )

    # Quick US Benchmark Test Button
    col_bench1, col_bench2 = st.columns([2, 3])
    with col_bench1:
        if st.button("🇺🇸 Load US Benchmark Test (Oyster Bay, NY)", use_container_width=True, key="btn_load_us_bench"):
            try:
                ev1_path = "docs/results/oyster_bay_ev1.json"
                ev2_path = "docs/results/oyster_bay_ev2.json"
                csv_path = "docs/results/oyster_bay_search_best_per_pair.csv"
                if os.path.exists(ev1_path) and os.path.exists(ev2_path):
                    st.session_state.events = [load_events_json(ev1_path)[0], load_events_json(ev2_path)[0]]
                if os.path.exists(csv_path):
                    st.session_state.search_results = pd.read_csv(csv_path)
                st.success("Loaded US Benchmark Test! Events & 328 candidate flight pairs loaded for Oyster Bay, NY.")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading US benchmark: {e}")
    with col_bench2:
        st.caption("Instantly loads the validated US reference recording events and flight results (40.876° N, -73.525° W).")

    col_p2_left, col_p2_right = st.columns([1, 1])

    with col_p2_left:
        st.subheader("📍 1. Geographic Location Circle")

        # Preset locations for quick selection
        preset_choice = st.selectbox(
            "Location Presets:",
            [
                "Bern, Switzerland (46.9480° N, 7.4474° E)",
                "Basel / EuroAirport (47.5898° N, 7.5336° E)",
                "Zurich Airport (47.4582° N, 8.5555° E)",
                "Geneva Airport (46.2370° N, 6.1092° E)",
                "Oyster Bay, NY (US Reference) (40.8765° N, -73.5250° W)",
                "SW Florida / Eagle Cam (26.7024° N, -81.8478° W)",
                "New York Metro (40.7128° N, -74.0060° W)",
                "Los Angeles Metro (34.0522° N, -118.2437° W)",
                "Chicago Metro (41.8781° N, -87.6298° W)",
                "Central USA / Kansas (39.8283° N, -98.5795° W)",
                "Custom Coordinates",
            ],
            index=0,
            key="p2_preset",
        )

        default_lat, default_lon = 46.9480, 7.4474
        if "Basel" in preset_choice:
            default_lat, default_lon = 47.5898, 7.5336
        elif "Zurich" in preset_choice:
            default_lat, default_lon = 47.4582, 8.5555
        elif "Geneva" in preset_choice:
            default_lat, default_lon = 46.2370, 6.1092
        elif "Oyster Bay" in preset_choice:
            default_lat, default_lon = 40.8765, -73.5250
        elif "Florida" in preset_choice:
            default_lat, default_lon = 26.7024, -81.8478
        elif "New York Metro" in preset_choice:
            default_lat, default_lon = 40.7128, -74.0060
        elif "Los Angeles" in preset_choice:
            default_lat, default_lon = 34.0522, -118.2437
        elif "Chicago" in preset_choice:
            default_lat, default_lon = 41.8781, -87.6298
        elif "Central USA" in preset_choice:
            default_lat, default_lon = 39.8283, -98.5795

        col_loc1, col_loc2, col_loc3 = st.columns(3)
        with col_loc1:
            center_lat = st.number_input("Center Latitude (°)", min_value=-90.0, max_value=90.0, value=default_lat, format="%.4f", key="p2_lat")
        with col_loc2:
            center_lon = st.number_input("Center Longitude (°)", min_value=-180.0, max_value=180.0, value=default_lon, format="%.4f", key="p2_lon")
        with col_loc3:
            radius_km = st.slider("Search Radius (km)", min_value=5.0, max_value=120.0, value=25.0, step=5.0, key="p2_radius")

        # Calculate bounding box from circle
        lat_min, lat_max, lon_min, lon_max = circle_to_bbox(center_lat, center_lon, radius_km)
        st.caption(f"Bounding Box: Lat `{lat_min:.3f}°` to `{lat_max:.3f}°`, Lon `{lon_min:.3f}°` to `{lon_max:.3f}°`")

    with col_p2_right:
        st.subheader("📅 2. Date & Time Range")
        search_date = st.date_input("Recording Date (UTC):", value=date(2026, 11, 2), key="p2_date")
        
        search_mode = st.radio(
            "Search Matching Mode:",
            [
                "Two-Event Relative Search (Uses time gap between Event 1 and 2)",
                "Single-Event with Known Time (Uses exact heard UTC time)",
            ],
            index=0,
            key="p2_search_mode",
        )

        t_heard_unix_val = None
        if "Single-Event" in search_mode:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                event_time_utc = st.time_input("Heard Time (UTC):", value=time(14, 36, 10), key="p2_time_utc")
            with col_t2:
                # Calculate unix timestamp
                dt_combined = datetime.combine(search_date, event_time_utc).replace(tzinfo=timezone.utc)
                t_heard_unix_val = dt_combined.timestamp()
                st.caption(f"Unix timestamp: `{t_heard_unix_val:.0f}` s")

    # ADS-B Day Index Selection
    st.markdown("---")
    st.subheader("✈️ 3. ADS-B Flight Data Index (`day.npz`)")

    col_idx1, col_idx2 = st.columns([3, 1])
    with col_idx1:
        local_day_files = [str(p) for p in Path(".").glob("*day*.npz")]
        if "day.npz" not in local_day_files and Path("day.npz").exists():
            local_day_files.insert(0, "day.npz")

        selected_day_file = st.selectbox(
            "Select Day Index File:",
            options=local_day_files if local_day_files else ["day.npz"],
            key="p2_day_sel",
        )
    with col_idx2:
        st.write("")
        st.write("")
        if st.button("Load Day Index", key="btn_load_day", use_container_width=True):
            if os.path.exists(selected_day_file):
                try:
                    with st.spinner("Loading Day Index..."):
                        st.session_state.day_index = DayIndex.load(selected_day_file)
                        st.session_state.day_source = selected_day_file
                        st.success(f"Loaded: `{selected_day_file}`")
                except Exception as e:
                    st.error(f"Error loading index: {e}")
            else:
                st.warning(f"File `{selected_day_file}` does not exist yet. You can fetch it below.")

    idx: DayIndex | None = st.session_state.day_index
    if idx is not None:
        st.success(f"Loaded Index: **{len(idx)} flight segments**, **{len(idx.t)} points**, origin at `{idx.lat0:.4f}°, {idx.lon0:.4f}°`")
    else:
        st.info("Tip: If you haven't fetched the day index yet, run: `uv run dopplertrack fetch YYYY-MM-DD day.npz --bbox 45.5 48.0 5.5 10.5` in terminal.")

    # Live Radar API (Today's Real-time traffic)
    st.markdown("---")
    st.subheader("📡 Live ADS-B Radar (Today's Real-Time Traffic API)")
    st.markdown(
        "For recordings made **today** (where GitHub daily archives are not yet published), "
        "query live transmitting flights in this circle via the free **adsb.lol / OpenSky Network** API."
    )
    col_live1, col_live2, col_live3 = st.columns([2, 1, 1])
    with col_live1:
        live_prop_only = st.checkbox("Filter: Propeller / Turboprop / Helicopter only", value=False, key="live_prop_only")
    with col_live2:
        query_live_btn = st.button("📡 Scan Live Flights in Circle", use_container_width=True, key="btn_query_live")
    with col_live3:
        if st.button("⏱️ Set Time to Now (UTC)", use_container_width=True, key="btn_set_time_now"):
            st.session_state.p2_time_utc = datetime.now(timezone.utc).time()
            st.rerun()

    if query_live_btn:
        with st.spinner(f"Querying live ADS-B traffic inside {radius_km:.0f} km around ({center_lat:.4f}, {center_lon:.4f})..."):
            from dopplertrack.live import fetch_live_flights_in_circle
            live_res = fetch_live_flights_in_circle(center_lat, center_lon, radius_km=radius_km, prop_only=live_prop_only)
            st.session_state.live_df = live_res
            if not live_res.empty:
                st.success(f"Found **{len(live_res)} live aircraft** currently in the circle!")
            else:
                st.warning("No active aircraft detected inside the circle right now.")

    live_df: pd.DataFrame | None = st.session_state.get("live_df")
    if live_df is not None and not live_df.empty:
        st.write("**Active Aircraft in Circle Right Now:**")
        cols_live = ["callsign", "reg", "type", "alt_m", "alt_ft", "speed_kmh", "speed_kts", "dist_km", "icao"]
        disp_live = [c for c in cols_live if c in live_df.columns]
        st.dataframe(live_df[disp_live], use_container_width=True)

    # Event Selection for Matching
    st.markdown("---")
    st.subheader("🎯 4. Events to Match & Search Tolerances")

    if not st.session_state.events:
        st.warning("⚠️ No events loaded! Please detect or fit events in Part 1 or ensure `detected.json` exists.")
    else:
        col_ev_sel1, col_ev_sel2, col_ev_sel3 = st.columns(3)
        with col_ev_sel1:
            ev_names = [f"Event #{i+1} (CPA {ev.t_cpa:.1f}s, f0={ev.f0:.1f}Hz, v={ev.v:.0f}m/s)" for i, ev in enumerate(st.session_state.events)]
            ev1_idx = st.selectbox("First Event (ev1):", options=list(range(len(ev_names))), index=0, format_func=lambda i: ev_names[i], key="p2_ev1_idx")
            ev1_chosen = st.session_state.events[ev1_idx]

        with col_ev_sel2:
            if "Single-Event" not in search_mode:
                ev2_default = min(1, len(ev_names) - 1)
                ev2_idx = st.selectbox("Second Event (ev2):", options=list(range(len(ev_names))), index=ev2_default, format_func=lambda i: ev_names[i], key="p2_ev2_idx")
                ev2_chosen = st.session_state.events[ev2_idx]
            else:
                st.write("")
                st.info("Single-event mode active (using exact heard time).")
                ev2_chosen = None

        with col_ev_sel3:
            v_tol = st.slider("Speed gate tolerance (±%)", min_value=15, max_value=60, value=35, step=5, key="p2_vtol") / 100.0
            tol_s = st.slider("Heard gap tolerance (seconds)", min_value=3, max_value=30, value=10, step=1, key="p2_tols")

        # RUN SEARCH BUTTON
        st.write("")
        run_search_btn = st.button("🚀 Scan Circle for Matching Flights", type="primary", use_container_width=True, key="btn_run_p2_search")

        if run_search_btn:
            if st.session_state.spectrogram is None:
                st.error("Spectrogram is not loaded! Please load or process a spectrogram in Part 1.")
            elif idx is None:
                st.error("Day Index is not loaded! Please load a valid day.npz file above.")
            else:
                bbox_tuple = (lat_min, lat_max, lon_min, lon_max)
                with st.spinner(f"Scanning flights inside {radius_km:.0f} km circle around ({center_lat:.3f}, {center_lon:.3f})..."):
                    try:
                        res_df = run_adsb_search(
                            sp=st.session_state.spectrogram,
                            ev1=ev1_chosen,
                            ev2=ev2_chosen,
                            idx=idx,
                            bbox=bbox_tuple,
                            t_heard_unix=t_heard_unix_val,
                            v_tol=v_tol,
                            tol_s=float(tol_s),
                            workers=1,  # Safe for Windows
                        )
                        st.session_state.search_results = res_df
                        if not res_df.empty:
                            st.success(f"Scan complete! Found **{len(res_df)} candidate points** across flight matches.")
                        else:
                            st.warning("Scan finished but 0 candidate points passed the gates. Try widening the speed tolerance or search radius.")
                    except Exception as e:
                        st.error(f"Search error: {e}")

    # ====================================================================================
    # RESULTS VISUALIZATION & MAP
    # ====================================================================================
    res_df: pd.DataFrame | None = st.session_state.search_results
    has_results = res_df is not None and not res_df.empty
    has_live = live_df is not None and not live_df.empty

    if has_results or has_live:
        st.markdown("---")
        st.header("🏆 Candidate Matches & Geolocation Map")

        # Map display
        fig_map = plot_geolocation_map(
            results_df=res_df,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            idx=st.session_state.day_index,
            live_df=live_df,
            top_k=25,
            title=f"DopplerTrack Geolocation — {radius_km:.0f} km Circle Scan",
        )
        st.plotly_chart(fig_map, use_container_width=True)

        if has_results:
            # Top result highlight card
            top_row = res_df.iloc[0]
            top_lat = float(top_row["lat"])
            top_lon = float(top_row["lon"])
            gmaps_url = f"https://www.google.com/maps/search/?api=1&query={top_lat},{top_lon}"

            st.subheader("🥇 Top Ranked Observer Location")
            card_col1, card_col2, card_col3, card_col4, card_col5 = st.columns(5)
            card_col1.metric("Latitude", f"{top_lat:.4f}°")
            card_col2.metric("Longitude", f"{top_lon:.4f}°")
            card_col3.metric("Elevation", f"{top_row.get('elev', 0):.0f} m")
            card_col4.metric("Score", f"{top_row.get('score', 0):.2f}")
            with card_col5:
                st.write("")
                st.link_button("📍 Open in Google Maps", gmaps_url, use_container_width=True)

            # Results table
            st.subheader("📊 Candidate Rankings Table")
            cols_to_show = ["gidx", "lat", "lon", "elev", "score", "icao1", "icao2", "v1", "d1", "v2", "d2", "shift"]
            display_cols = [c for c in cols_to_show if c in res_df.columns]
            
            df_display = res_df[display_cols].copy()
            if "v1" in df_display:
                df_display["v1 (km/h)"] = (df_display["v1"] * 3.6).round(1)
            if "v2" in df_display:
                df_display["v2 (km/h)"] = (df_display["v2"] * 3.6).round(1)
            if "d1" in df_display:
                df_display["d1 (m)"] = df_display["d1"].round(0)
            if "d2" in df_display:
                df_display["d2 (m)"] = df_display["d2"].round(0)

            st.dataframe(df_display.head(50), use_container_width=True)

            # Download CSV
            csv_bytes = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Results as CSV (`results.csv`)",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
                use_container_width=True,
            )
