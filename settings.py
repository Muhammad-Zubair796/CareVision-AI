import cv2
import time
import json
import os
import threading
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

PKT = timezone(timedelta(hours=5))

def get_pkt_time():
    return datetime.now(PKT).strftime("%A, %d %B %Y | %I:%M:%S %p PKT")

CONFIG_FILE = "hospital_config.json"

def load_db():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "name": "City Care Hospital",
        "active_dept": 0,
        "departments": [
            {"id": 0, "name": "Emergency Department (ER)", "total_chairs": 12, "source": "clinic_waiting.mp4"},
            {"id": 1, "name": "General OPD & Pediatrics Lounge", "total_chairs": 25, "source": "busy_lounge.mp4"},
            {"id": 2, "name": "Terminal Waiting Hall", "total_chairs": 30, "source": "cam3_terminal.mp4"}
        ]
    }

def save_db():
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(hospital_db, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")

hospital_db = load_db()

live_stats = {
    d["id"]: {"total": 0, "seated": 0, "standing": 0, "empty": d["total_chairs"]}
    for d in hospital_db["departments"]
}

class FreshFrameReader:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.02)
                continue
            self.frame = frame
            time.sleep(0.02)

    def update_source(self, new_source):
        if new_source != self.source:
            self.source = new_source
            old_cap = self.cap
            self.cap = cv2.VideoCapture(new_source)
            try:
                old_cap.release()
            except Exception:
                pass

    def stop(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

readers = {d["id"]: FreshFrameReader(d["source"]) for d in hospital_db["departments"]}

def navbar():
    cur_id = hospital_db.get("active_dept", 0)
    links = "".join([
        f'<a href="/live/{d["id"]}" class="px-3.5 py-2 rounded-lg text-xs font-semibold border border-slate-700 '
        f'{"bg-blue-600 text-white" if d["id"] == cur_id else "bg-slate-800 text-slate-300 hover:bg-slate-700"}">'
        f'{d["name"]} ({d["total_chairs"]})</a>\n'
        for d in hospital_db["departments"]
    ])
    return f"""<header class="flex flex-wrap justify-between items-center mb-6 pb-4 border-b border-slate-800 gap-4">
        <div><h1 class="text-xl font-bold text-blue-400">🏥 {hospital_db['name']}</h1>
        <p class="text-xs text-slate-400">Enterprise AI Patient Flow & Seating Capacity SaaS</p></div>
        <div class="flex flex-wrap items-center gap-2">
            <a href="/" class="bg-indigo-600 hover:bg-indigo-500 text-white px-3.5 py-2 rounded-lg text-xs font-bold shadow-lg">🖥️ Central Command Overview</a>
            {links}
            <a href="/add_dept" class="bg-emerald-600 hover:bg-emerald-500 text-white px-3.5 py-2 rounded-lg text-xs font-bold">➕ Add Dept</a>
            <a href="/settings" class="bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3.5 py-2 rounded-lg text-xs font-bold text-slate-300">⚙️ Settings</a>
        </div></header>"""

settings_router = APIRouter()

@settings_router.get("/settings", response_class=HTMLResponse)
def settings_page():
    inputs = ""
    for d in hospital_db["departments"]:
        dept_id = d["id"]
        inputs += f"""
        <div class="bg-slate-850 p-4 rounded-xl border border-slate-800 space-y-3">
            <div class="flex justify-between items-center pb-2 border-b border-slate-800">
                <input type="text" name="name_{dept_id}" value="{d['name']}" class="bg-slate-800 border border-slate-700 p-2 rounded-lg text-sm text-blue-400 font-bold w-2/3" required>
                <a href="/delete_dept/{dept_id}" onclick="return confirm('Are you sure you want to delete {d['name']}?');" 
                   class="bg-red-950 hover:bg-red-900 border border-red-800 text-red-400 text-xs px-3 py-1.5 rounded-lg font-bold transition">
                   🗑️ Delete Dept
                </a>
            </div>
            <div class="grid grid-cols-2 gap-3">
                <div>
                    <label class="text-xs text-slate-400 font-semibold">Total Chairs</label>
                    <input type="number" name="c_{dept_id}" value="{d['total_chairs']}" min="1" class="w-full bg-slate-800 border border-slate-700 p-2 rounded-lg text-sm text-white mt-1" required>
                </div>
                <div>
                    <label class="text-xs text-slate-400 font-semibold">Video Source / RTSP Stream</label>
                    <input type="text" name="s_{dept_id}" value="{d['source']}" class="w-full bg-slate-800 border border-slate-700 p-2 rounded-lg text-sm text-white mt-1" required>
                </div>
            </div>
        </div>
        """

    return f"""<!DOCTYPE html><html><head><title>Hospital Configuration</title><script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-slate-950 text-white min-h-screen p-6">{navbar()}
        <div class="max-w-2xl mx-auto bg-slate-900 border border-slate-800 p-8 rounded-2xl shadow-2xl space-y-6">
            <div class="flex justify-between items-center border-b border-slate-800 pb-4">
                <h1 class="text-xl font-bold text-blue-400">⚙️ Hospital Configuration</h1>
                <a href="/add_dept" class="bg-emerald-600 hover:bg-emerald-500 text-xs font-bold px-3 py-2 rounded-lg text-white">➕ Add New Ward</a>
            </div>
            <form action="/save_settings" method="post" class="space-y-4">
                <div>
                    <label class="text-xs text-slate-300 font-semibold">Hospital Name</label>
                    <input type="text" name="hname" value="{hospital_db['name']}" class="w-full bg-slate-800 border border-slate-700 p-3 rounded-lg text-white font-bold text-sm mt-1" required>
                </div>
                <div class="space-y-3">
                    <label class="text-xs text-slate-400 font-semibold uppercase tracking-wider">Configured Wards & Cameras</label>
                    {inputs}
                </div>
                <button type="submit" class="w-full bg-blue-600 hover:bg-blue-500 py-3.5 rounded-xl font-bold text-sm shadow-lg transition">
                    💾 Save All Settings & Return to Overview →
                </button>
            </form>
        </div></body></html>"""

@settings_router.post("/save_settings")
async def save_settings(request: Request):
    form = await request.form()
    hospital_db["name"] = form.get("hname", hospital_db["name"])

    for d in hospital_db["departments"]:
        dept_id = d["id"]
        if f"name_{dept_id}" in form:
            d["name"] = str(form[f"name_{dept_id}"]).strip()
        if f"c_{dept_id}" in form:
            try:
                d["total_chairs"] = int(form[f"c_{dept_id}"])
            except ValueError:
                pass
        if f"s_{dept_id}" in form:
            new_source = str(form[f"s_{dept_id}"]).strip()
            if new_source and new_source != d["source"]:
                d["source"] = new_source
                if dept_id in readers:
                    readers[dept_id].update_source(new_source)
                else:
                    readers[dept_id] = FreshFrameReader(new_source)

    save_db()
    return RedirectResponse(url="/", status_code=303)

@settings_router.get("/delete_dept/{dept_id}")
def delete_department(dept_id: int):
    hospital_db["departments"] = [d for d in hospital_db["departments"] if d["id"] != dept_id]
    if dept_id in readers:
        readers[dept_id].stop()
        del readers[dept_id]
    if dept_id in live_stats:
        del live_stats[dept_id]

    if hospital_db.get("active_dept") == dept_id:
        hospital_db["active_dept"] = hospital_db["departments"][0]["id"] if hospital_db["departments"] else 0

    save_db()
    return RedirectResponse(url="/settings", status_code=303)

@settings_router.get("/add_dept", response_class=HTMLResponse)
def add_dept_page():
    return f"""<!DOCTYPE html><html><head><title>Add Department</title><script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-slate-950 text-white min-h-screen p-6">{navbar()}
        <div class="max-w-lg mx-auto bg-slate-900 border border-slate-800 p-8 rounded-2xl shadow-2xl space-y-4">
            <h1 class="text-xl font-bold text-blue-400">➕ Add Waiting Area / CCTV</h1>
            <form action="/save_dept" method="post" class="space-y-4">
                <div>
                    <label class="block text-xs font-semibold mb-1">Ward Name</label>
                    <input type="text" name="dname" placeholder="e.g. Trauma Care" class="w-full bg-slate-800 border border-slate-700 p-3 rounded-lg text-white text-sm" required>
                </div>
                <div>
                    <label class="block text-xs font-semibold mb-1">Total Chairs</label>
                    <input type="number" name="chairs" value="20" min="1" class="w-full bg-slate-800 border border-slate-700 p-3 rounded-lg text-white text-sm" required>
                </div>
                <div>
                    <label class="block text-xs font-semibold mb-1">CCTV RTSP IP or Video File</label>
                    <input type="text" name="src" placeholder="rtsp://admin:pass@ip:554/live or file.mp4" class="w-full bg-slate-800 border border-slate-700 p-3 rounded-lg text-white text-sm" required>
                </div>
                <button type="submit" class="w-full bg-emerald-600 hover:bg-emerald-500 py-3 rounded-xl font-bold transition">Connect & Add Department →</button>
            </form>
        </div></body></html>"""

@settings_router.post("/save_dept")
def save_new_department(dname: str = Form(...), chairs: int = Form(...), src: str = Form(...)):
    existing_ids = [d["id"] for d in hospital_db["departments"]]
    new_id = (max(existing_ids) + 1) if existing_ids else 0
    hospital_db["departments"].append({"id": new_id, "name": dname, "total_chairs": chairs, "source": src})
    readers[new_id] = FreshFrameReader(src)
    live_stats[new_id] = {"total": 0, "seated": 0, "standing": 0, "empty": chairs}
    save_db()
    return RedirectResponse(url=f"/live/{new_id}", status_code=303)
