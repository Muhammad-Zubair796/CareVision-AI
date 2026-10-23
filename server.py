import cv2
import time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from ultralytics import YOLO

from settings import (
    settings_router,
    hospital_db,
    live_stats,
    readers,
    navbar,
    get_pkt_time
)
from analytics import analytics_router, record_live_snapshot

app = FastAPI(title="Hospital Vision SaaS")
app.include_router(settings_router)
app.include_router(analytics_router)

model = YOLO("yolo11n-pose.pt")

# ==============================================================================
# VISION INFERENCE & POSE CLASSIFIER
# ==============================================================================
def is_sitting(kpts, box):
    w, h = box[2] - box[0], box[3] - box[1]
    if (h / (w + 1e-6)) < 1.4: return True
    if kpts is None or len(kpts) < 17: return (h / (w + 1e-6)) < 2.0
    l_hip_y, r_hip_y = kpts[11][1], kpts[12][1]
    l_knee_y, r_knee_y = kpts[13][1], kpts[14][1]
    l_sh_y, r_sh_y = kpts[5][1], kpts[6][1]
    if l_knee_y == 0 and r_knee_y == 0: return True
    hip_y = max(l_hip_y, r_hip_y)
    knee_y = max(l_knee_y, r_knee_y)
    sh_y = max(l_sh_y, r_sh_y)
    return knee_y == 0 or abs(knee_y - hip_y) < (0.65 * abs(hip_y - sh_y))

def draw_hud(frame_h, stats, dept_name):
    hud = np.zeros((frame_h, 480, 3), dtype=np.uint8)
    hud[:] = (16, 22, 34)

    cv2.putText(hud, "HOSPITAL PATIENT FLOW", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(hud, dept_name.upper()[:28], (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1)

    cv2.rectangle(hud, (25, 85), (455, 140), (26, 34, 50), -1)
    cv2.putText(hud, "TOTAL PATIENTS IN WAITING AREA", (35, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 190), 1)
    cv2.putText(hud, f"{stats['total']} Patients", (35, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.rectangle(hud, (25, 155), (230, 230), (26, 34, 50), -1)
    cv2.putText(hud, "PATIENTS SEATED", (35, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 170, 190), 1)
    cv2.putText(hud, str(stats["seated"]), (35, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 130), 2)

    cv2.rectangle(hud, (250, 155), (455, 230), (26, 34, 50), -1)
    cv2.putText(hud, "PATIENTS STANDING", (260, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 170, 190), 1)
    cv2.putText(hud, str(stats["standing"]), (260, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 2)

    cv2.rectangle(hud, (25, 245), (455, 315), (26, 34, 50), -1)
    cv2.putText(hud, f"TOTAL CHAIRS IN AREA: {stats['total_chairs']}", (35, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    empty_col = (0, 255, 120) if stats["empty"] > 0 else (0, 0, 255)
    cv2.putText(hud, f"Available Empty Chairs: {stats['empty']}", (35, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.52, empty_col, 2)

    cv2.putText(hud, "ADMIN RECOMMENDATION", (25, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if stats["total"] > stats["total_chairs"]:
        deficit = stats["total"] - stats["total_chairs"]
        box_bg, title, sub = (20, 20, 90), f"ALERT: DEFICIT OF {deficit} CHAIRS", f"Deploy +{deficit + 2} chairs immediately."
    elif stats["empty"] > 5:
        box_bg, title, sub = (25, 60, 30), "CAPACITY OPTIMAL (AMPLE SEATING)", f"{stats['empty']} empty chairs available. All accommodated."
    else:
        box_bg, title, sub = (70, 45, 20), "CAPACITY BALANCED", f"Seating matches patient flow ({stats['empty']} open)."

    cv2.rectangle(hud, (25, 360), (455, 435), box_bg, -1)
    cv2.putText(hud, title, (35, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
    cv2.putText(hud, sub, (35, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
    return hud

def generate_stream(dept_id: int):
    dept_matches = [d for d in hospital_db["departments"] if d["id"] == dept_id]
    if not dept_matches:
        return
    dept = dept_matches[0]
    reader = readers.get(dept_id)
    if not reader:
        return

    cached_boxes, frame_idx = [], 0

    while True:
        raw = reader.frame
        if raw is None:
            time.sleep(0.04)
            continue

        frame = cv2.resize(raw.copy(), (800, 520))
        h, w, _ = frame.shape
        frame_idx += 1

        if frame_idx % 2 == 0 or not cached_boxes:
            res = model.predict(frame, conf=0.20, imgsz=640, verbose=False)
            cached_boxes, seated, standing = [], 0, 0
            if res[0].boxes is not None:
                boxes = res[0].boxes.xyxy.cpu().numpy()
                kpts = res[0].keypoints.xy.cpu().numpy() if res[0].keypoints is not None else []
                for i, box in enumerate(boxes):
                    sit = is_sitting(kpts[i] if i < len(kpts) else None, box)
                    if sit:
                        seated += 1
                        lbl, col = "Seated Patient", (0, 255, 0)
                    else:
                        standing += 1
                        lbl, col = "Standing Patient", (0, 140, 255)
                    cached_boxes.append((box, lbl, col))

            empty = max(0, dept["total_chairs"] - seated)
            stats = {
                "total": seated + standing,
                "seated": seated,
                "standing": standing,
                "empty": empty,
                "total_chairs": dept["total_chairs"]
            }
            live_stats[dept_id] = stats

        for box, lbl, col in cached_boxes:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, 2)
            cv2.putText(frame, lbl, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2)

        hud = draw_hud(h, stats, dept["name"])
        composite = np.hstack((frame, hud))
        _, buf = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.035)

# ==============================================================================
# CENTRAL COMMAND OVERVIEW
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def command_overview():
    now_pkt = get_pkt_time()

    # Camera Cards (aspect-ratio 1280/520 prevents cropping)
    cam_cards = ""
    for d in hospital_db["departments"]:
        cam_cards += f"""
        <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg flex flex-col justify-between">
            <div class="p-3 bg-slate-850 border-b border-slate-800 flex justify-between items-center">
                <span class="font-bold text-xs text-blue-400">{d['name']}</span>
                <span class="text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800 px-2 py-0.5 rounded font-bold">● LIVE</span>
            </div>
            <a href="/live/{d['id']}" class="block relative group bg-black overflow-hidden" style="aspect-ratio: 1280 / 520;">
                <img src="/stream/{d['id']}" class="w-full h-full object-contain">
                <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition">
                    <span class="bg-blue-600 text-white text-xs px-3 py-1.5 rounded-lg font-bold shadow">🔍 Inspect Full Screen</span>
                </div>
            </a>
            <div class="p-3 bg-slate-900 border-t border-slate-800 flex justify-between items-center text-xs">
                <span class="text-slate-400">Chairs: <b class="text-white">{d['total_chairs']}</b></span>
                <a href="/live/{d['id']}" class="text-blue-400 hover:underline font-bold">Inspect Stream →</a>
            </div>
        </div>
        """
# Multi-Ward Table Rows
    rows = ""
    for d in hospital_db["departments"]:
        s = live_stats.get(d["id"], {"total": 0, "seated": 0, "standing": 0, "empty": d["total_chairs"]})
        rows += f"""<tr class="border-b border-slate-800 hover:bg-slate-850">
            <td class="p-4 font-bold text-white">{d['name']}</td>
            <td class="p-4 text-center">{d['total_chairs']}</td>
            <td id="row_p_{d['id']}" class="p-4 text-center font-bold text-lg text-white">{s['total']}</td>
            <td id="row_s_{d['id']}" class="p-4 text-center text-emerald-400 font-bold">{s['seated']}</td>
            <td id="row_st_{d['id']}" class="p-4 text-center text-amber-400 font-bold">{s['standing']}</td>
            <td id="row_e_{d['id']}" class="p-4 text-center text-blue-400 font-bold">{s['empty']}</td>
            <td id="row_badge_{d['id']}" class="p-4 text-center">
                <span class="px-2 py-1 bg-emerald-950 border border-emerald-700 text-emerald-400 rounded text-xs font-bold">🟢 AMPLE SEATS</span>
            </td>
            <td class="p-4 text-center"><a href="/live/{d['id']}" class="bg-blue-600 hover:bg-blue-500 text-white text-xs px-3 py-1.5 rounded font-bold">Inspect →</a></td>
        </tr>"""

    return f"""<!DOCTYPE html><html><head><title>Central Command</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <script>
        let liveDonutChart = null;
        let timelineBarChart = null;
        let trendLineChart = null;
        let currentRange = "30d";

        document.addEventListener("DOMContentLoaded", function () {{
            // 1. LIVE DONUT CHART
            liveDonutChart = new ApexCharts(document.querySelector("#chart-live-pie"), {{
                series: [1, 1, 1],
                chart: {{ type: 'donut', height: 280, background: 'transparent' }},
                labels: ['Seated Patients', 'Standing Patients', 'Available Chairs'],
                colors: ['#10B981', '#F59E0B', '#3B82F6'],
                stroke: {{ show: false }},
                dataLabels: {{ enabled: false }},
                legend: {{ position: 'bottom', labels: {{ colors: '#94A3B8' }} }},
                plotOptions: {{
                    pie: {{
                        donut: {{
                            size: '72%',
                            labels: {{
                                show: true,
                                name: {{ show: true, color: '#94A3B8', fontSize: '12px' }},
                                value: {{ show: true, color: '#FFFFFF', fontSize: '22px', fontWeight: 700 }},
                                total: {{
                                    show: true,
                                    label: 'LIVE PATIENTS',
                                    color: '#94A3B8',
                                    formatter: function (w) {{ return (w.globals.series[0] + w.globals.series[1]); }}
                                }}
                            }}
                        }}
                    }}
                }},
                theme: {{ mode: 'dark' }}
            }});
            liveDonutChart.render();

            // 2. TIMELINE BAR CHART
            timelineBarChart = new ApexCharts(document.querySelector("#chart-timeline-bar"), {{
                series: [{{ name: 'Total', data: [] }}, {{ name: 'Seated', data: [] }}, {{ name: 'Standing', data: [] }}],
                chart: {{ type: 'bar', height: 280, background: 'transparent', toolbar: {{ show: false }} }},
                plotOptions: {{ bar: {{ columnWidth: '55%', borderRadius: 4 }} }},
                colors: ['#6366F1', '#10B981', '#F59E0B'],
                dataLabels: {{ enabled: false }},
                stroke: {{ show: true, width: 2, colors: ['transparent'] }},
                xaxis: {{ categories: [], labels: {{ style: {{ colors: '#94A3B8', fontSize: '11px' }}, rotate: -25 }} }},
                yaxis: {{ labels: {{ style: {{ colors: '#94A3B8' }} }} }},
                grid: {{ borderColor: '#1E293B' }},
                theme: {{ mode: 'dark' }},
                legend: {{ position: 'top', horizontalAlign: 'right', labels: {{ colors: '#94A3B8' }} }}
            }});
            timelineBarChart.render();

            // 3. CONTINUOUS TREND LINE / AREA CHART (Rises and Falls over days & weeks)
            trendLineChart = new ApexCharts(document.querySelector("#chart-trend-line"), {{
                series: [
                    {{ name: 'Peak Surge Flow', data: [] }},
                    {{ name: 'Average Baseline Flow', data: [] }}
                ],
                chart: {{
                    type: 'area',
                    height: 300,
                    background: 'transparent',
                    toolbar: {{ show: false }},
                    zoom: {{ enabled: false }}
                }},
                colors: ['#06B6D4', '#6366F1'],
                dataLabels: {{ enabled: false }},
                stroke: {{ curve: 'smooth', width: 3 }},
                fill: {{
                    type: 'gradient',
                    gradient: {{ shadeIntensity: 1, opacityFrom: 0.45, opacityTo: 0.05, stops: [0, 90, 100] }}
                }},
                xaxis: {{
                    categories: [],
                    labels: {{ style: {{ colors: '#94A3B8', fontSize: '11px' }}, rotate: -25 }},
                    axisBorder: {{ color: '#334155' }}
                }},
                yaxis: {{
                    title: {{ text: 'Patients Count', style: {{ color: '#94A3B8' }} }},
                    labels: {{ style: {{ colors: '#94A3B8' }} }}
                }},
                grid: {{ borderColor: '#1E293B' }},
                theme: {{ mode: 'dark' }},
                legend: {{ position: 'top', labels: {{ colors: '#94A3B8' }} }},
                tooltip: {{
                    theme: 'dark',
                    y: {{ formatter: function (val) {{ return val + " Patients"; }} }}
                }}
            }});
            trendLineChart.render();

            fetchHistoricalTimeline();
            fetchTrendData("30d");
            setInterval(fetchHistoricalTimeline, 10000);
        }});

        // Switch time range filter (24h, 7d, 30d)
        function switchRange(range) {{
            currentRange = range;
            ['btn_24h', 'btn_7d', 'btn_30d'].forEach(id => {{
                document.getElementById(id).className = "px-3 py-1 rounded-lg text-xs font-bold border border-slate-700 bg-slate-800 text-slate-400 hover:bg-slate-700 transition";
            }});
            document.getElementById('btn_' + range).className = "px-3 py-1 rounded-lg text-xs font-bold border border-cyan-500 bg-cyan-600 text-white shadow-lg transition";
            fetchTrendData(range);
        }}

        async function fetchTrendData(range) {{
            try {{
                const res = await fetch('/api/trend_data?range_type=' + range);
                const data = await res.json();
                if (trendLineChart) {{
                    trendLineChart.updateOptions({{ xaxis: {{ categories: data.categories }} }});
                    trendLineChart.updateSeries([
                        {{ name: 'Peak Surge Flow', data: data.totals }},
                        {{ name: 'Average Baseline Flow', data: data.seated }}
                    ]);
                }}
            }} catch(e) {{}}
        }}

        async function fetchHistoricalTimeline() {{
            try {{
                const res = await fetch('/api/analytics_history');
                const data = await res.json();
                if (timelineBarChart && data.categories.length > 0) {{
                    timelineBarChart.updateOptions({{ xaxis: {{ categories: data.categories }} }});
                    timelineBarChart.updateSeries([
                        {{ name: 'Total', data: data.totals }},
                        {{ name: 'Seated', data: data.seated }},
                        {{ name: 'Standing', data: data.standing }}
                    ]);
                    document.getElementById('peak_badge').innerText = "🔥 Peak Surge: " + data.peak_info;
                }}
            }} catch(e) {{}}
        }}

        async function updateStats() {{
            try {{
                const res = await fetch('/api/central_stats');
                const data = await res.json();
                
                document.getElementById('sys_clock').innerText = data.clock;
                document.getElementById('tot_patients').innerText = data.tot_p;
                document.getElementById('tot_chairs').innerText = data.tot_c;
                document.getElementById('tot_seated').innerText = data.tot_s;
                document.getElementById('tot_standing').innerText = data.tot_st;
                document.getElementById('tot_empty').innerText = data.tot_e;

                if (liveDonutChart) {{
                    liveDonutChart.updateSeries([data.tot_s, data.tot_st, data.tot_e]);
                }}

                data.depts.forEach(d => {{
                    const elP = document.getElementById('row_p_' + d.id);
                    const elS = document.getElementById('row_s_' + d.id);
                    const elSt = document.getElementById('row_st_' + d.id);
                    const elE = document.getElementById('row_e_' + d.id);
                    const elBadge = document.getElementById('row_badge_' + d.id);
                    
                    if (elP) elP.innerText = d.total;
                    if (elS) elS.innerText = d.seated;
                    if (elSt) elSt.innerText = d.standing;
                    if (elE) elE.innerText = d.empty;
                    if (elBadge) {{
                        if (d.total > d.total_chairs) {{
                            elBadge.innerHTML = '<span class="px-2 py-1 bg-red-950 border border-red-700 text-red-400 rounded text-xs font-bold">🔴 OVERCROWDED</span>';
                        }} else {{
                            elBadge.innerHTML = '<span class="px-2 py-1 bg-emerald-950 border border-emerald-700 text-emerald-400 rounded text-xs font-bold">🟢 AMPLE SEATS</span>';
                        }}
                    }}
                }});
            }} catch(err) {{}}
        }}
        setInterval(updateStats, 1000);
    </script></head>
    <body class="bg-slate-950 text-white min-h-screen p-6">{navbar()}
        <main class="max-w-6xl mx-auto space-y-6">
            <!-- HEADER -->
            <div class="flex justify-between items-center bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-lg">
                <div><h2 class="text-lg font-bold">Central Waiting Area Command Overview</h2><p class="text-xs text-slate-400">Live operational monitoring across all hospital departments.</p></div>
                <div class="text-right"><span class="text-xs text-slate-400">SYSTEM CLOCK</span><div id="sys_clock" class="text-sm font-bold text-emerald-400 font-mono">{now_pkt}</div></div>
            </div>
<!-- CAMERAS -->
            <div>
                <h3 class="text-sm font-bold text-slate-300 mb-3 flex items-center space-x-2">
                    <span>📹 Live Waiting Area Camera Windows</span>
                    <span class="text-xs text-slate-500 font-normal">(Click any camera to inspect full-screen)</span>
                </h3>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
                    {cam_cards}
                </div>
            </div>

            <!-- 5 KPI METRICS -->
            <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl text-center"><span class="text-xs text-slate-400 font-semibold uppercase">Total Patients</span><div id="tot_patients" class="text-3xl font-extrabold text-white mt-1">--</div></div>
                <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl text-center"><span class="text-xs text-slate-400 font-semibold uppercase">Total Chairs</span><div id="tot_chairs" class="text-3xl font-extrabold text-blue-400 mt-1">--</div></div>
                <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl text-center"><span class="text-xs text-slate-400 font-semibold uppercase">Seated</span><div id="tot_seated" class="text-3xl font-extrabold text-emerald-400 mt-1">--</div></div>
                <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl text-center"><span class="text-xs text-slate-400 font-semibold uppercase">Standing</span><div id="tot_standing" class="text-3xl font-extrabold text-amber-400 mt-1">--</div></div>
                <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl text-center"><span class="text-xs text-slate-400 font-semibold uppercase">Empty Chairs</span><div id="tot_empty" class="text-3xl font-extrabold text-blue-300 mt-1">--</div></div>
            </div>

            <!-- ROW 1: LIVE DONUT & RECENT TIMELINE BAR -->
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col justify-between">
                    <div>
                        <div class="flex justify-between items-center mb-1">
                            <h4 class="font-bold text-sm text-slate-200">Live Occupancy & Patient Ratio</h4>
                            <span class="text-[10px] bg-blue-950 text-blue-400 border border-blue-800 px-2 py-0.5 rounded font-bold">LIVE TELEMETRY</span>
                        </div>
                        <p class="text-xs text-slate-400 mb-2">Real-time split between seated, standing, & available seats.</p>
                    </div>
                    <div id="chart-live-pie" class="flex justify-center items-center py-2"></div>
                </div>

                <div class="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col justify-between">
                    <div>
                        <div class="flex flex-wrap justify-between items-center gap-2 mb-1">
                            <div>
                                <h4 class="font-bold text-sm text-slate-200">Patient Inflow & Peak Hours Timeline</h4>
                                <p class="text-xs text-slate-400">Continuous historical load tracking identifying daily peak surge periods.</p>
                            </div>
                            <span id="peak_badge" class="text-xs bg-amber-950 border border-amber-700 text-amber-300 px-3 py-1 rounded-lg font-bold shadow animate-pulse">
                                🔥 Calculating Peak...
                            </span>
                        </div>
                    </div>
                    <div id="chart-timeline-bar" class="w-full"></div>
                </div>
            </div>

            <!-- ROW 2: HISTORICAL MULTI-DAY / 30-DAY CONTINUOUS TREND LINE GRAPH -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-2xl space-y-3">
                <div class="flex flex-wrap justify-between items-center gap-3 border-b border-slate-800 pb-3">
                    <div>
                        <h4 class="font-bold text-sm text-slate-200 flex items-center gap-2">
                            <span>📈 Multi-Day Surge Trend & Influx Curves</span>
                            <span class="text-[10px] bg-cyan-950 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded font-bold">HISTORICAL INTELLIGENCE</span>
                        </h4>
                        <p class="text-xs text-slate-400">Track how hospital traffic rose and fell over hours, days, and the past month.</p>
                    </div>
                    <!-- Range Filter Buttons -->
                    <div class="flex items-center gap-2">
                        <button id="btn_24h" onclick="switchRange('24h')" class="px-3 py-1 rounded-lg text-xs font-bold border border-slate-700 bg-slate-800 text-slate-400 hover:bg-slate-700 transition">24 Hours</button>
                        <button id="btn_7d" onclick="switchRange('7d')" class="px-3 py-1 rounded-lg text-xs font-bold border border-slate-700 bg-slate-800 text-slate-400 hover:bg-slate-700 transition">7 Days</button>
                        <button id="btn_30d" onclick="switchRange('30d')" class="px-3 py-1 rounded-lg text-xs font-bold border border-cyan-500 bg-cyan-600 text-white shadow-lg transition">Past 30 Days</button>
                    </div>
                </div>
                <div id="chart-trend-line" class="w-full"></div>
            </div>

            <!-- TELEMETRY TABLE -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
                <div class="p-4 border-b border-slate-800 font-bold text-sm text-slate-200">Live Multi-Ward Telemetry Table</div>
                <table class="w-full text-left text-sm"><thead class="bg-slate-850 text-slate-400 text-xs uppercase border-b border-slate-800"><tr>
                    <th class="p-4">Waiting Ward</th><th class="p-4 text-center">Chairs</th><th class="p-4 text-center">Patients</th><th class="p-4 text-center">Seated</th><th class="p-4 text-center">Standing</th><th class="p-4 text-center">Empty</th><th class="p-4 text-center">Status</th><th class="p-4 text-center">Action</th></tr></thead>
                    <tbody>{rows}</tbody></table>
            </div>
        </main>
    </body></html>"""

@app.get("/api/central_stats")
async def api_central_stats():
    now_pkt = get_pkt_time()
    tot_p = sum(s.get("total", 0) for s in live_stats.values())
    tot_c = sum(d["total_chairs"] for d in hospital_db["departments"])
    tot_s = sum(s.get("seated", 0) for s in live_stats.values())
    tot_st = sum(s.get("standing", 0) for s in live_stats.values())
    tot_e = sum(s.get("empty", 0) for s in live_stats.values())

    record_live_snapshot(tot_p, tot_s, tot_st)

    depts_data = []
    for d in hospital_db["departments"]:
        s = live_stats.get(d["id"], {"total": 0, "seated": 0, "standing": 0, "empty": d["total_chairs"]})
        depts_data.append({
            "id": d["id"],
            "name": d["name"],
            "total_chairs": d["total_chairs"],
            "total": s["total"],
            "seated": s["seated"],
            "standing": s["standing"],
            "empty": s["empty"]
        })

    return JSONResponse({
        "clock": now_pkt,
        "tot_p": tot_p,
        "tot_c": tot_c,
        "tot_s": tot_s,
        "tot_st": tot_st,
        "tot_e": tot_e,
        "depts": depts_data
    })

@app.get("/live/{dept_id}", response_class=HTMLResponse)
def live_page(dept_id: int):
    hospital_db["active_dept"] = dept_id
    dept_matches = [d for d in hospital_db["departments"] if d["id"] == dept_id]
    dept = dept_matches[0] if dept_matches else hospital_db["departments"][0]

    return f"""<!DOCTYPE html><html><head><title>Live: {dept['name']}</title><script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-slate-950 text-white min-h-screen p-6">{navbar()}
        <main class="max-w-6xl mx-auto bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
            <div class="p-3 bg-slate-850 border-b border-slate-800 flex justify-between items-center text-xs text-slate-400">
                <span>Active Stream: <b class="text-white">{dept['name']}</b> ({dept['total_chairs']} Total Chairs)</span>
                <span class="text-emerald-400 font-bold">● ZERO-LATENCY REAL-TIME STREAM</span>
            </div>
            <img src="/stream/{dept['id']}" class="w-full h-auto">
        </main></body></html>"""

@app.get("/stream/{dept_id}")
def stream(dept_id: int):
    return StreamingResponse(generate_stream(dept_id), media_type="multipart/x-mixed-replace; boundary=frame")
