# ai_superapp.py
import os, json, time, re, math, hashlib
from flask import Flask, request, jsonify, render_template_string, Response, send_file
import requests
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()

# ========== CONFIG ==========
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")   # <-- paste your Groq key
GROQ_MODEL   = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# OCR.space fallback (optional; default key is rate-limited)
OCRSPACE_API_KEY = os.getenv("OCRSPACE_API_KEY", "helloworld")
OCRSPACE_ENDPOINT = "https://api.ocr.space/parse/image"

# Data dirs
DATA_DIR   = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
IMG_DIR    = os.path.join(DATA_DIR, "images")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Defaults
MAX_TURNS       = 20
MAX_DOC_CHARS   = 200_000
CHUNK_SIZE      = 1500
TOP_K           = 4
CHATLIST_FILE   = os.path.join(DATA_DIR, "chats.json")  # keeps list of chat folders

PERSONAS = {
    "default": "You are a helpful assistant.",
    "tutor":   "You are a patient tutor. Explain step-by-step and add a tiny quiz when helpful.",
    "coach":   "You are a motivational coach. Be concise and encouraging.",
    "coder":   "You are a senior Python engineer. Provide runnable code and short notes."
}

# ========== HELPERS (storage) ==========
def _chat_paths(chat_id):
    safe = re.sub(r"[^a-zA-Z0-9_\-]+","_", chat_id or "default")
    mem   = os.path.join(DATA_DIR, f"{safe}_memory.json")
    facts = os.path.join(DATA_DIR, f"{safe}_facts.json")
    docs  = os.path.join(DATA_DIR, f"{safe}_docs.json")
    return mem, facts, docs

def _safe_load(path, empty):
    if not os.path.exists(path): return empty
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except: return empty

def _safe_save(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _load_chatlist():
    return _safe_load(CHATLIST_FILE, {"chats": ["default"]})

def _save_chatlist(obj):
    _safe_save(CHATLIST_FILE, obj)

def load_memory(chat_id):
    mem_path, _, _ = _chat_paths(chat_id)
    return _safe_load(mem_path, {"messages": []})

def save_memory(chat_id, mem):
    mem_path, _, _ = _chat_paths(chat_id)
    _safe_save(mem_path, mem)

def append_memory(chat_id, role, content):
    mem = load_memory(chat_id)
    mem.setdefault("messages", [])
    mem["messages"].append({"role": role, "content": content, "ts": int(time.time())})
    ua = [m for m in mem["messages"] if m["role"] in ("user","assistant")]
    if len(ua) > MAX_TURNS * 2:
        mem["messages"] = mem["messages"][-(MAX_TURNS * 2):]
    save_memory(chat_id, mem)

def clear_memory(chat_id):
    save_memory(chat_id, {"messages": []})

def load_facts(chat_id):
    _, facts_path, _ = _chat_paths(chat_id)
    return _safe_load(facts_path, {"facts": []})

def save_facts(chat_id, fx):
    _, facts_path, _ = _chat_paths(chat_id)
    _safe_save(facts_path, fx)

def add_fact(chat_id, fact_text):
    fx = load_facts(chat_id)
    fx.setdefault("facts", [])
    fact_text = (fact_text or "").strip()
    if fact_text and fact_text not in fx["facts"]:
        fx["facts"].append(fact_text)
        save_facts(chat_id, fx)

def load_docs(chat_id):
    _, _, docs_path = _chat_paths(chat_id)
    return _safe_load(docs_path, {"docs": {}})

def save_docs(chat_id, d):
    _, _, docs_path = _chat_paths(chat_id)
    _safe_save(docs_path, d)

# ========= PDF text + OCR =========
def extract_pdf_text_native(path):
    try:
        reader = PdfReader(path)
        out = []
        for page in reader.pages:
            out.append(page.extract_text() or "")
        text = "\n".join(out)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()[:MAX_DOC_CHARS]
    except: return ""

def ocrspace_pdf(path, language="eng"):
    try:
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f)}
            data = {
                "apikey": OCRSPACE_API_KEY,
                "language": language,   # eng, hin, mar
                "isOverlayRequired": False,
                "OCREngine": 2
            }
            r = requests.post(OCRSPACE_ENDPOINT, files=files, data=data, timeout=180)
        if r.status_code != 200:
            return ""
        js = r.json()
        if not js.get("IsErroredOnProcessing"):
            parsed = [res.get("ParsedText","") for res in js.get("ParsedResults", [])]
            return "\n".join(parsed).strip()[:MAX_DOC_CHARS]
        return ""
    except: return ""

def extract_pdf_text_with_ocr(path):
    text = extract_pdf_text_native(path)
    if len(text) >= 200:
        return text, False
    for lang in ("eng", "hin", "mar"):
        text = ocrspace_pdf(path, lang)
        if len(text) >= 100:
            return text, True
    return text, True if text else False

# ========= RAG helpers =========
def chunk_text(s, size=CHUNK_SIZE):
    return [s[i:i+size] for i in range(0, len(s), size)]

def score_chunk(chunk, query):
    def norm(t):
        t = t.lower()
        t = re.sub(r"[^a-z0-9\u0900-\u097F]+", " ", t)
        return [w for w in t.split() if len(w) > 2]
    qw = set(norm(query))
    if not qw: return 0
    cw = norm(chunk)
    if not cw: return 0
    hit = sum(1 for w in cw if w in qw)
    return hit / math.sqrt(1 + len(cw)/200)

def select_top_chunks(text, query, k=TOP_K):
    if not text.strip(): return []
    ch = chunk_text(text)
    scored = sorted([(score_chunk(c, query), c) for c in ch], key=lambda x: x[0], reverse=True)
    return [c for (sc,c) in scored[:k] if sc>0] or ch[:min(k,len(ch))]

# ========= LLM call + Offline fallback =========
def call_groq(messages, temperature=0.6, max_tokens=512):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": messages, "temperature": float(temperature), "max_tokens": int(max_tokens)}
    try:
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            d = r.json()
            return True, d["choices"][0]["message"]["content"].strip()
        else:
            return False, f"{r.status_code} {r.reason}: {r.text[:300]}"
    except Exception as e:
        return False, f"REQUEST_ERROR: {e}"

def offline_reply(history_text):
    last = ""
    for line in reversed(history_text.splitlines()):
        if line.startswith("User:"):
            last = line[5:].strip()
            break
    if not last:
        return "I'm offline. How can I help?"
    if any(x in last.lower() for x in ["hello","hi","hey","namaste","नमस्ते","नमस्कार"]):
        return "Hi! (offline mode) — I can still chat briefly. Ask me something simple."
    if last.endswith("?"):
        return "I’m offline, but here’s a short thought: " + last.rstrip("?") + " — consider key points, examples, and a conclusion."
    return f"(offline) You said: “{last}”. I suggest breaking this into steps: goal → key info → action → review."

def build_messages(chat_id, persona_key="default"):
    mem = load_memory(chat_id)
    facts = load_facts(chat_id).get("facts", [])
    persona = PERSONAS.get(persona_key, PERSONAS["default"])
    facts_text = ("Long-term facts about the user:\n- " + "\n- ".join(facts) + "\n\n") if facts else ""
    sys = (f"{persona}\nUse prior turns AND long-term facts for consistency.\n" +
           "If the user says 'remember that ...', extract the core fact.\n" + facts_text)
    msgs = [{"role":"system","content":sys}]
    for m in mem["messages"]:
        if m["role"] in ("user","assistant"):
            msgs.append({"role":m["role"],"content":m["content"]})
    return msgs

# ========= Flask App / UI =========
app = Flask(__name__)

INDEX = """
<!doctype html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AI SuperApp</title>
<style>
:root{
  --bg:#0f172a; --card:#0b1220; --fg:#e2e8f0; --stroke:#1f2937; --pill:#334155; --u:#1e293b; --b:#111827;
}
.light:root{ --bg:#f8fafc; --card:#ffffff; --fg:#0f172a; --stroke:#e5e7eb; --pill:#d1d5db; --u:#f1f5f9; --b:#ffffff; }
.pastel:root{ --bg:#f7f6ff; --card:#fffaff; --fg:#2b2d42; --stroke:#e9e7ff; --pill:#d6d4ff; --u:#e9f5ff; --b:#fff3f3; }

body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: var(--bg); color: var(--fg);}
.wrap { max-width: 980px; margin: 0 auto; padding: 14px; }
h2 { margin: 0 0 10px; }
.grid { display:grid; grid-template-columns: 1fr 1fr; gap:10px; }
.card { background:var(--card); border:1px solid var(--stroke); border-radius:14px; padding:10px; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap;}
select,input[type=number],input[type=text]{ padding:8px; border-radius:10px; border:1px solid var(--pill); background:var(--card); color:var(--fg);}
button{ padding:8px 12px; border-radius:12px; border:1px solid var(--pill); background:var(--b); color:var(--fg); cursor:pointer; }
.pill{ padding:6px 10px; border-radius:999px; border:1px solid var(--pill); background:var(--card); color:var(--fg); cursor:pointer; }
.chat{ height:50vh; overflow:auto; background:var(--card); border:1px solid var(--stroke); border-radius:14px; padding:10px;}
.crow{ display:flex; gap:6px; margin:6px 0; align-items:flex-end;}
.u .bub{ background:var(--u); border-radius:14px; padding:10px; max-width:75%; }
.b .bub{ background:var(--b); border:1px solid var(--stroke); border-radius:14px; padding:10px; max-width:75%;}
.rem{ font-size:12px; border-style:dashed;}
.controls{ display:grid; grid-template-columns:1fr auto; gap:8px; }
textarea{ height:62px; padding:10px; border-radius:12px; border:1px solid var(--pill); background:var(--card); color:var(--fg);}
small{ color:#9ca3af;}
img{ max-width:100%; border-radius:12px; border:1px solid var(--stroke); }

/* docs list */
#doclist{ min-width:260px; max-height:160px; overflow:auto; border:1px solid var(--pill); border-radius:10px; padding:6px; }
.docrow{ display:flex; align-items:center; justify-content:space-between; gap:8px; margin:4px 0; }
.docleft{ display:flex; gap:8px; align-items:center; }
.docleft span{ word-break: break-all; }
</style>
</head><body>
<div class="wrap">
  <h2>🤖 AI SuperApp</h2>

  <div class="grid">
    <div class="card">
      <div class="row">
        <label>Chat folder:</label>
        <select id="chat"></select>
        <input id="newchat" placeholder="new chat name"/>
        <button onclick="createChat()">Create</button>
        <button onclick="deleteChat()">Delete</button>
      </div>
      <div class="row">
        <label>Persona</label>
        <select id="persona">
          <option>default</option><option>tutor</option><option>coach</option><option>coder</option>
        </select>
        <label>Temperature</label><input id="temp" type="number" step="0.1" min="0" max="1.5" value="0.6"/>
        <label>Voice</label>
        <select id="lang">
          <option value="auto">Auto</option>
          <option value="en-IN">English (India)</option>
          <option value="hi-IN">Hindi</option>
          <option value="mr-IN">Marathi</option>
        </select>
        <label>Theme</label>
        <select id="theme"><option value="dark">Dark</option><option value="light">Light</option><option value="pastel">Pastel</option></select>
        <label>Mode</label>
        <select id="mode"><option value="online">Online</option><option value="offline">Offline</option></select>
      </div>

      <div id="chatbox" class="chat"></div>
      <div class="controls">
        <textarea id="msg" placeholder="Type message…"></textarea>
        <div class="row">
          <button onclick="send()">Send</button>
          <button id="micBtn" onclick="toggleMic()">🎤</button>
        </div>
      </div>

      <div class="row" style="margin-top:6px;">
        <label><input type="checkbox" id="remember"> Remember this message as a fact</label>
        <button class="pill" onclick="showFacts()">👁️ Facts</button>
        <button class="pill" onclick="clearFacts()">🗑️ Clear Facts</button>
        <a class="pill" href="#" id="exportTxt" target="_blank">⬇️ Export TXT</a>
        <a class="pill" href="#" id="exportJson" target="_blank">⬇️ Export JSON</a>
      </div>
    </div>

    <div class="card">
      <h3>📄 PDF Q&A</h3>
      <div class="row">
        <input type="file" id="pdf" accept="application/pdf"/>
        <button onclick="uploadPdf()">Upload</button>
        <small id="upinfo"></small>
      </div>

      <!-- NEW DOCS UI with per-item delete -->
      <div class="row">
        <label>Docs</label>
        <div id="doclist"></div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <button onclick="delDocs()" title="Delete checked">🗑️ Delete checked</button>
          <button onclick="delDocsAll()" title="Delete ALL docs">🗑️🚨 Delete ALL</button>
        </div>
        <a id="dldoc" class="pill" href="#" target="_blank">⬇️ Download Doc Text</a>
      </div>

      <div class="row">
        <input id="docq" placeholder="Ask about selected docs or leave empty to search all"/>
        <button onclick="askDoc()">Ask PDF</button>
      </div>
      <div id="doca" style="white-space:pre-wrap; margin-top:6px;"></div>
    </div>
  </div>

  <div class="card" style="margin-top:10px;">
    <h3>🎨 Image Generation (free)</h3>
    <div class="row">
      <input id="imgp" placeholder="Describe the image you want"/>
      <button onclick="genImg()">Generate</button>
    </div>
    <div id="imgarea" style="margin-top:8px;"></div>
    <small>Powered by Pollinations (no key). Images are proxied for reliability.</small>
  </div>
</div>

<script>
let voices=[], recog=null, recognizing=false;

function applyTheme(){
  const theme = localStorage.getItem('theme') || 'dark';
  document.documentElement.className = (theme==='light'?'light':(theme==='pastel'?'pastel':'')); 
  document.getElementById('theme').value = theme;
}
applyTheme();
document.getElementById('theme').addEventListener('change', ()=>{
  const t = document.getElementById('theme').value;
  localStorage.setItem('theme', t);
  applyTheme();
});

function loadVoices(){ voices = speechSynthesis.getVoices(); }
if (typeof speechSynthesis!=='undefined'){ loadVoices(); speechSynthesis.onvoiceschanged = loadVoices; }

function pickVoice(langPref){
  if(!voices.length) return null;
  if(langPref && langPref!=='auto'){
    let v = voices.find(v=> (v.lang||'').toLowerCase()===langPref.toLowerCase());
    if(v) return v;
    const base = langPref.split('-')[0];
    v = voices.find(v=> (v.lang||'').toLowerCase().startsWith(base));
    if(v) return v;
  }
  const prefs = ["hi-IN","hi","mr-IN","mr","en-IN","en"];
  for(const p of prefs){
    let v = voices.find(v=> (v.lang||'').toLowerCase()===p.toLowerCase());
    if(v) return v;
    const base = p.split('-')[0];
    v = voices.find(v=> (v.lang||'').toLowerCase().startsWith(base));
    if(v) return v;
  }
  return voices[0]||null;
}
function speak(text){
  try{
    const pref = document.getElementById('lang').value || 'auto';
    const u = new SpeechSynthesisUtterance(text);
    const v = pickVoice(pref);
    if(v) u.voice = v;
    u.lang = v ? v.lang : (pref==='auto' ? 'en-IN' : pref);
    speechSynthesis.cancel(); speechSynthesis.speak(u);
  }catch(e){}
}

function chatRow(role, text, idx){
  const wrap = document.createElement('div'); wrap.className = 'crow ' + (role==='user'?'u':'b');
  const bub  = document.createElement('div'); bub.className = 'bub'; bub.textContent = text;
  wrap.appendChild(bub);
  if(role==='user'){
    const btn = document.createElement('button'); btn.className='pill rem'; btn.textContent='Remember';
    btn.onclick = async ()=>{ await fetch('/remember_msg', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ chat:curChat(), text })}); alert('Saved as fact.'); };
    wrap.appendChild(btn);
  }
  return wrap;
}

async function refreshChat(){
  const chat = curChat();
  const r = await fetch('/state?chat='+encodeURIComponent(chat));
  const data = await r.json();
  const box = document.getElementById('chatbox'); box.innerHTML='';
  data.forEach(p=>{
    if(p[0]) box.appendChild(chatRow('user',p[0]));
    if(p[1]) box.appendChild(chatRow('assistant',p[1]));
  });
  box.scrollTop = box.scrollHeight;
  const bubbles = box.querySelectorAll('.b .bub'); if(bubbles.length){ speak(bubbles[bubbles.length-1].textContent); }
  document.getElementById('exportTxt').href = '/export?chat='+encodeURIComponent(chat);
  document.getElementById('exportJson').href= '/export_json?chat='+encodeURIComponent(chat);
}

function curChat(){ return document.getElementById('chat').value || 'default'; }

async function loadChats(){
  const r = await fetch('/chats'); const js = await r.json();
  const sel = document.getElementById('chat'); sel.innerHTML='';
  (js.chats||[]).forEach(name=>{
    const opt = document.createElement('option'); opt.value=name; opt.textContent=name; sel.appendChild(opt);
  });
  if(!(js.chats||[]).length){ sel.innerHTML='<option>default</option>'; }
  refreshChat(); loadDocs();
}
async function createChat(){
  const name = (document.getElementById('newchat').value||'').trim(); if(!name) return;
  await fetch('/chats', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ action:'create', name })});
  document.getElementById('newchat').value='';
  loadChats();
}
async function deleteChat(){
  const chat = curChat();
  if(!confirm('Delete chat '+chat+'? This will remove its memory, facts, and docs.')) return;
  await fetch('/chats', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ action:'delete', name:chat })});
  loadChats();
}

async function send(){
  const text = (document.getElementById('msg').value||'').trim(); if(!text) return;
  const persona = document.getElementById('persona').value;
  const temperature = parseFloat(document.getElementById('temp').value||'0.6');
  const remember = document.getElementById('remember').checked;
  const mode = document.getElementById('mode').value;
  document.getElementById('msg').value='';
  await fetch('/send', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ chat:curChat(), text, persona, temperature, remember, mode })});
  refreshChat();
}

function toggleMic(){
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) { alert('No speech recognition in this browser.'); return; }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!recog){
    recog = new SR();
    const pref = document.getElementById('lang').value||'en-IN';
    recog.lang = pref==='auto' ? 'en-IN' : pref; recog.interimResults=false; recog.maxAlternatives=1;
    recog.onresult = (e)=>{ document.getElementById('msg').value = e.results[0][0].transcript; send(); };
    recog.onerror = ()=>{ recognizing=false; document.getElementById('micBtn').textContent='🎤'; };
    recog.onend   = ()=>{ recognizing=false; document.getElementById('micBtn').textContent='🎤'; };
  }
  const pref = document.getElementById('lang').value||'en-IN'; recog.lang = pref==='auto'?'en-IN':pref;
  if(!recognizing){ recognizing=true; document.getElementById('micBtn').textContent='⏹️'; recog.start(); }
  else{ recognizing=false; document.getElementById('micBtn').textContent='🎤'; recog.stop(); }
}

// Keep-alive ping
setInterval(()=>{ fetch('/health').catch(()=>{}); }, 25000);

/* ======= DOCS UI with per-item delete ======= */
function docRow(name){
  const row = document.createElement('div');
  row.className = 'docrow';

  const left = document.createElement('div');
  left.className = 'docleft';

  const chk = document.createElement('input');
  chk.type = 'checkbox';
  chk.value = name;
  chk.className = 'docchk';

  const label = document.createElement('span');
  label.textContent = name;

  left.appendChild(chk);
  left.appendChild(label);

  const actions = document.createElement('div');

  const dl = document.createElement('a');
  dl.className = 'pill';
  dl.href = '/doctext?chat=' + encodeURIComponent(curChat()) + '&name=' + encodeURIComponent(name);
  dl.target = '_blank';
  dl.textContent = '⬇️';

  const del = document.createElement('button');
  del.textContent = '🗑️';
  del.title = 'Delete this PDF';
  del.onclick = async ()=>{
    if(!confirm('Delete "'+name+'"?')) return;
    await fetch('/docs', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ chat:curChat(), action:'delete', names:[name] })
    });
    loadDocs();
  };

  actions.appendChild(dl);
  actions.appendChild(del);

  row.appendChild(left);
  row.appendChild(actions);
  return row;
}

async function loadDocs(){
  const r = await fetch('/docs?chat=' + encodeURIComponent(curChat()));
  const js = await r.json();
  const list = document.getElementById('doclist');
  list.innerHTML = '';
  (js.names || []).forEach(n => list.appendChild(docRow(n)));

  const first = (js.names || [])[0];
  document.getElementById('dldoc').href = first
    ? '/doctext?chat=' + encodeURIComponent(curChat()) + '&name=' + encodeURIComponent(first)
    : '#';
}

async function delDocs(){
  const checked = Array.from(document.querySelectorAll('.docchk:checked')).map(x => x.value);
  if(!checked.length) { alert('No PDFs selected. Tick the boxes first.'); return; }
  if(!confirm('Delete selected PDFs?')) return;
  await fetch('/docs', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ chat:curChat(), action:'delete', names: checked })
  });
  loadDocs();
}

async function delDocsAll(){
  const all = Array.from(document.querySelectorAll('.docchk')).map(x => x.value);
  if(!all.length){ alert('No PDFs uploaded.'); return; }
  if(!confirm('Delete ALL PDFs in this chat?')) return;
  await fetch('/docs', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ chat:curChat(), action:'delete', names: all })
  });
  loadDocs();
}

/* ======= PDFs actions ======= */
async function uploadPdf(){
  const f = document.getElementById('pdf').files[0]; const info = document.getElementById('upinfo');
  if(!f){ info.textContent="Choose a PDF first."; return; }
  const fd = new FormData(); fd.append('file', f); fd.append('chat', curChat());
  info.textContent="Uploading… (OCR fallback if needed)";
  try{
    const r = await fetch('/upload', {method:'POST', body: fd}); const js = await r.json();
    info.textContent = js.ok ? ("Uploaded: "+js.name+(js.ocr?" (OCR used)":"")) : ("Error: "+js.error);
  }catch(e){ info.textContent = "Upload failed: "+e; }
  loadDocs();
}

async function askDoc(){
  const boxes = Array.from(document.querySelectorAll('.docchk:checked')).map(x=>x.value);
  const q = (document.getElementById('docq').value||'').trim();
  const out = document.getElementById('doca');
  out.textContent = "Thinking…";
  const body = { chat:curChat(), names:boxes, question:q };
  const r = await fetch('/askdoc', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const js = await r.json(); out.textContent = js.answer || js.error || "No answer.";
  // speak result
  try{
    const bubbles = document.querySelectorAll('.b .bub');
    const last = js.answer || "";
    if(last) speak(last);
  }catch(e){}
}

/* ======= Images ======= */
async function genImg(){
  const p = (document.getElementById('imgp').value||'').trim(); if(!p) return;
  const r = await fetch('/genimg', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ prompt:p })});
  const js = await r.json();
  const div = document.getElementById('imgarea');
  if(js.ok){
    div.innerHTML = '<a href="'+js.url+'" target="_blank"><img src="'+js.url+'"/></a>';
  }else{
    div.innerHTML = '<small>'+js.error+'</small>';
  }
}

/* ======= Init ======= */
async function init(){
  const r = await fetch('/chats'); const js = await r.json();
  const sel = document.getElementById('chat'); sel.innerHTML='';
  (js.chats||[]).forEach(n=>{ const o=document.createElement('option'); o.value=n; o.textContent=n; sel.appendChild(o); });
  refreshChat(); loadDocs();
}
document.getElementById('chat').addEventListener('change', ()=>{ refreshChat(); loadDocs(); });

// Keep-alive ping for Pydroid
setInterval(()=>{ fetch('/health').catch(()=>{}); }, 25000);

init();
</script>
</body></html>
"""

# ========= Routes =========
@app.route("/")
def index():
    return render_template_string(INDEX)

@app.route("/health")
def health():
    return "ok", 200

# --- Chat folders ---
@app.route("/chats", methods=["GET","POST"])
def chats():
    cfg = _load_chatlist()
    if request.method == "POST":
        data = request.get_json(force=True)
        act = data.get("action")
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"ok":False,"error":"name required"}), 400
        if act=="create":
            if name not in cfg["chats"]:
                cfg["chats"].append(name)
                _save_chatlist(cfg)
        elif act=="delete":
            if name=="default":
                return jsonify({"ok":False,"error":"cannot delete default"}), 400
            if name in cfg["chats"]:
                cfg["chats"].remove(name); _save_chatlist(cfg)
                # remove files
                mem,facts,docs = _chat_paths(name)
                for p in [mem,facts,docs]:
                    try: os.remove(p)
                    except: pass
        return jsonify({"ok":True,"chats":cfg["chats"]})
    return jsonify(cfg)

@app.route("/state")
def state():
    chat = request.args.get("chat","default")
    mem = load_memory(chat)["messages"]
    out=[]; pair=[None,None]
    for m in mem:
        if m["role"]=="user":
            if pair!=[None,None]: out.append(pair); pair=[None,None]
            pair[0]=m["content"]
        elif m["role"]=="assistant":
            if pair[0] is None: out.append([None,m["content"]])
            else: pair[1]=m["content"]; out.append(pair); pair=[None,None]
    if pair!=[None,None]: out.append(pair)
    return jsonify(out)

@app.route("/remember_msg", methods=["POST"])
def remember_msg():
    data = request.get_json(force=True)
    chat = data.get("chat","default")
    text = (data.get("text") or "").strip()
    if not text: return jsonify({"ok":False,"error":"empty"}), 400
    add_fact(chat, text)
    return jsonify({"ok":True})

@app.route("/send", methods=["POST"])
def send():
    data = request.get_json(force=True)
    chat = data.get("chat","default")
    text = (data.get("text") or "").strip()
    persona = data.get("persona") or "default"
    temperature = float(data.get("temperature") or 0.6)
    remember = bool(data.get("remember"))
    mode = data.get("mode","online")

    if not text: return jsonify({"ok":True})

    if remember: add_fact(chat, text)

    banned = ["password:", "credit card", "cvv", "otp:"]
    if any(b in text.lower() for b in banned):
        append_memory(chat,"user",text)
        bot = "Sorry, I can’t help with sensitive secrets."
        append_memory(chat,"assistant",bot)
        return jsonify({"ok":True,"bot":bot})

    append_memory(chat,"user",text)
    msgs = build_messages(chat, persona)

    if mode=="offline":
        hist = " ".join([f"{m['role'].title()}: {m['content']}" for m in load_memory(chat)["messages"][-8:]])
        bot = offline_reply(hist)
        append_memory(chat,"assistant",bot)
        return jsonify({"ok":True,"bot":bot})

    ok, out = call_groq(msgs, temperature=temperature)
    if not ok:
        hist = " ".join([f"{m['role'].title()}: {m['content']}" for m in load_memory(chat)["messages"][-8:]])
        bot = offline_reply(hist) + "  [Used offline fallback]"
    else:
        bot = out
    append_memory(chat,"assistant",bot)
    return jsonify({"ok":True,"bot":bot})

@app.route("/facts")
def facts():
    chat = request.args.get("chat","default")
    fx = load_facts(chat).get("facts",[])
    return "Facts:\n- " + "\n- ".join(fx) if fx else "No facts yet."

@app.route("/clearfacts", methods=["POST"])
def clearfacts():
    chat = (request.get_json(silent=True) or {}).get("chat","default")
    save_facts(chat, {"facts":[]})
    return jsonify({"ok":True})

@app.route("/export")
def export_txt():
    chat = request.args.get("chat","default")
    mem = load_memory(chat)["messages"]
    lines = []
    for m in mem:
        if m["role"] in ("user","assistant"):
            lines.append(f"{m['role'].title()}: {m['content']}")
    txt = "\n".join(lines) if lines else "(empty)"
    return Response(txt, mimetype="text/plain; charset=utf-8")

@app.route("/export_json")
def export_json():
    chat = request.args.get("chat","default")
    mem = load_memory(chat)["messages"]
    payload = {"messages":[m for m in mem if m.get("role") in ("user","assistant")]}
    return Response(json.dumps(payload,ensure_ascii=False,indent=2), mimetype="application/json")

# ---- PDFs ----
@app.route("/docs", methods=["GET","POST"])
def docs_list():
    if request.method=="POST":
        data = request.get_json(force=True)
        chat = data.get("chat","default")
        action = data.get("action")
        if action=="delete":
            names = data.get("names",[])
            d = load_docs(chat)
            for n in names:
                d["docs"].pop(n, None)
            save_docs(chat, d)
            return jsonify({"ok":True})
        return jsonify({"ok":False,"error":"unknown action"}), 400
    chat = request.args.get("chat","default")
    d = load_docs(chat)
    return jsonify({"names": sorted(list(d["docs"].keys()))})

@app.route("/upload", methods=["POST"])
def upload_pdf():
    chat = request.form.get("chat","default")
    if "file" not in request.files: return jsonify({"ok":False,"error":"no file"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"ok":False,"error":"only PDF allowed"}), 400
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)

    native = extract_pdf_text_native(path)
    used_ocr = False
    text = native
    if len((text or "").strip()) < 200:
        text, used_ocr = extract_pdf_text_with_ocr(path)

    if not text:
        return jsonify({"ok":False,"error":"Failed to extract text (even with OCR)."}), 500

    d = load_docs(chat); d["docs"][f.filename] = text[:MAX_DOC_CHARS]; save_docs(chat, d)
    return jsonify({"ok":True, "name":f.filename, "ocr":used_ocr})

@app.route("/doctext")
def doctext():
    chat = request.args.get("chat","default")
    name = (request.args.get("name") or "").strip()
    d = load_docs(chat)
    if name not in d.get("docs",{}): return "Not found", 404
    txt = d["docs"][name]
    return Response(txt, mimetype="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename={name}.txt"})

@app.route("/askdoc", methods=["POST"])
def askdoc():
    data = request.get_json(force=True)
    chat = data.get("chat","default")
    question = (data.get("question") or "").strip()
    names = data.get("names") or []
    d = load_docs(chat)
    if not d.get("docs"): return jsonify({"error":"No documents uploaded yet."}), 400

    # Search in selected docs or all docs if none picked
    corpus = []
    if names:
        for n in names:
            if n in d["docs"]:
                corpus.append((n, d["docs"][n]))
    else:
        corpus = list(d["docs"].items())

    ranked_chunks = []
    for name, text in corpus:
        for c in select_top_chunks(text, question, k=TOP_K):
            ranked_chunks.append((name, c))
    ranked_chunks = ranked_chunks[:6] if ranked_chunks else []
    context = "\n\n".join([f"[{nm}] {ck}" for (nm,ck) in ranked_chunks]) if ranked_chunks else ""

    if not context:
        return jsonify({"answer":"Couldn't find relevant passages in the selected documents."})

    sys = "Answer ONLY using the provided document context. If not present, say you cannot find it in the document."
    msgs = [
        {"role":"system","content":sys},
        {"role":"user","content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely with citations like [docname]."}
    ]
    ok, out = call_groq(msgs, temperature=0.2, max_tokens=700)
    ans = out if ok else "(Error) " + out
    return jsonify({"answer": ans})

# ---- Image generation (Pollinations proxy) ----
@app.route("/genimg", methods=["POST"])
def genimg():
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok":False, "error":"Empty prompt"}), 400

    # ---------- Pollinations ----------
    try:
        url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?nologo=true&nofeed=true"
        r = requests.get(url, timeout=60)

        if r.status_code == 200 and len(r.content) > 5000:
            h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            fname = f"img_{h}.jpg"
            path = os.path.join(IMG_DIR, fname)
            with open(path, "wb") as f:
                f.write(r.content)
            return jsonify({"ok":True, "url": f"/img/{fname}"})
    except Exception as e:
        print("Pollinations failed:", e)

    # ---------- Stable Horde ----------
    try:
        submit = requests.post(
            "https://stablehorde.net/api/v2/generate/async",
            json={"prompt": prompt, "params": {"sampler_name":"k_euler"}, "nsfw": False},
            headers={"apikey": "0000000000"},
            timeout=30
        ).json()

        job_id = submit.get("id")
        if job_id:
            for _ in range(40):
                time.sleep(1)
                check = requests.get(
                    f"https://stablehorde.net/api/v2/generate/status/{job_id}"
                ).json()

                if check.get("done") and check.get("generations"):
                    img_b64 = check["generations"][0]["img"]

                    import base64
                    img_data = base64.b64decode(img_b64 + "===")

                    h = hashlib.sha256((prompt+"horde").encode()).hexdigest()[:16]
                    fname = f"img_{h}.png"
                    path = os.path.join(IMG_DIR, fname)

                    with open(path, "wb") as f:
                        f.write(img_data)

                    return jsonify({"ok":True, "url": f"/img/{fname}"})
    except Exception as e:
        print("Stable Horde failed:", e)

    # ---------- HuggingFace ----------
        # ---------- HuggingFace ----------
    try:
        HF_TOKEN = os.getenv("HF_TOKEN")

        if HF_TOKEN:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            hf_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"

            response = requests.post(
                hf_url,
                headers=headers,
                json={"inputs": prompt},
                timeout=120
            )

            # HF returns image as bytes OR error as JSON
            if response.status_code == 200 and response.headers.get("content-type","").startswith("image"):
                img_data = response.content

                h = hashlib.sha256((prompt+"hf").encode()).hexdigest()[:16]
                fname = f"img_{h}.png"
                path = os.path.join(IMG_DIR, fname)

                with open(path, "wb") as f:
                    f.write(img_data)

                return jsonify({"ok":True, "url": f"/img/{fname}"})
            else:
                print("HF returned non-image:", response.text[:200])

    except Exception as e:
        print("HF failed:", e)

    return jsonify({"ok":False, "error":"Image service temporarily unavailable"})
@app.route("/img/<name>")
def imgserve(name):
    path = os.path.join(IMG_DIR, name)
    if not os.path.exists(path):
        return "Not found", 404

    if name.endswith(".png"):
        mime = "image/png"
    elif name.endswith(".jpg"):
        mime = "image/jpeg"
    elif name.endswith(".webp"):
        mime = "image/webp"
    else:
        mime = "application/octet-stream"

    return send_file(path, mimetype=mime)

if __name__ == "__main__":
    if not GROQ_API_KEY.startswith("gsk_"):
        print("⚠️ Paste your Groq API key at the top (GROQ_API_KEY = 'gsk_...').")
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
