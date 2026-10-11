"""LIVE TOURNAMENT EXPLORER v3 — full viewer interaction + SF17 analysis.

  standings + cross table + pairing cards
  game viewer: numbered move table, keyboard arrows, |< < > >|, auto-play,
    sticky ply across refreshes, PGN download, flip
  ANALYZE toggle: neutral Stockfish 17.1 (depth 18, cached per game) →
    eval bar per ply, per-move loss coloring, mistake/blunder marks,
    turning-points list ("where the loser went wrong")
  live games: PV reconstruction from fastchess logs (legality-truncated)

Endpoints: /  /api/status  /api/pgn  /api/live  /api/analyze  /chess.js  /pieces/*
Stateless; analysis cached under TDIR/analysis/.
"""
import json, os, re, glob, sys, time
import chess as _chess
import chess.pgn as _chess_pgn
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

TDIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/cold-raid6/chess-audit/swiss_teams"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8077
WEB = "/srv/workspace/flychess/src/chess-lab/web"
SF17 = "/usr/games/stockfish"
AN_DEPTH = 18

RESULT_CP = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
ALL_PLAYERS = ["nQ", "r2_ep19", "r2_ep39", "r2_ep59", "r2_last", "r3_ep19", "r3_last"]

LEGEND = [
    ("nQ", "naked quarter-trained — hard routing, no head (champion of this event, 9.0/12)"),
    ("r2_ep19", "all 13 round-2 experts at epoch 19 (early in round-2 training)"),
    ("r2_ep39", "all 13 round-2 experts at epoch 39 (mid round-2) — tied nQ 9.0/12"),
    ("r2_ep59", "all 13 round-2 experts at epoch 59"),
    ("r2_last", "all 13 round-2 final checkpoints = nR2 as played (8.5/12)"),
    ("r3_ep19", "round-3 early snapshots (2.0/12)"),
    ("r3_last", "round-3 final checkpoints = nR3 (1.0/12 — round-3 confirmed destructive)"),
    ("nR2", "naked round-2 — hard routing, no head"),
    ("nR3", "naked round-3 — hard routing, no head"),
    ("sQ", "stacked quarter — head on 13 evals + HCE + domain gates, calibrated on quarter"),
    ("sR2", "stacked round-2 — head calibrated on round-2 evals"),
    ("sR3", "stacked round-3 — head calibrated on round-3 evals (tactics col = quarter fallback)"),
    ("e2850", "SF17.1 at nominal UCI_Elo 2850 — WARNING: miscalibrated anchor, actual strength far below label (double the blunder rate of the strong cluster; empirically club-level). sf8 is the trusted reference"),
    ("sf8", "full-strength Stockfish 8 reference build (CCRL 3223 class)"),
]

PV_TOK = re.compile(r" pv (\S+)")
SCORE = re.compile(r"score cp (-?\d+)")


def read_pairings():
    out = []
    for pgn in sorted(glob.glob(TDIR + "/*.pgn")):
        name = os.path.basename(pgn)[:-4]
        if "_vs_" not in name:
            continue
        a, b = name.split("_vs_")
        try:
            txt = open(pgn).read()
        except OSError:
            continue
        games = []
        for gtxt in txt.split("[Event")[1:]:
            wh = re.search(r"\[White .([^\"]+)", gtxt)
            bl = re.search(r"\[Black .([^\"]+)", gtxt)
            res = re.search(r"\[Result .([^\"]+)", gtxt)
            plies = re.search(r"\[PlyCount .(\d+)", gtxt)
            games.append({"white": wh.group(1) if wh else "?",
                          "black": bl.group(1) if bl else "?",
                          "result": res.group(1) if res else "*",
                          "plies": int(plies.group(1)) if plies else 0})
        log = pgn.replace(".pgn", ".log")
        active, last_eval, started = False, None, None
        try:
            ltxt = open(log, errors="ignore").read()
            m = re.findall(r"Started game (\d+) of 6 \((\w+) vs (\w+)\)", ltxt)
            if m and int(m[-1][0]) > len(games):
                active, started = True, m[-1]
            evs = SCORE.findall(ltxt)
            if evs:
                last_eval = int(evs[-1])
        except OSError:
            pass
        out.append({"a": a, "b": b, "games": games, "active": active,
                    "current": started, "last_eval": last_eval})
    return out


def live_state(log_path):
    toks, last, evs = [], None, []
    try:
        lines = open(log_path, errors="ignore").read().splitlines()
    except OSError:
        return toks, None
    for line in lines:
        if "Started game" in line:
            toks, last, evs = [], None, []
            continue
        m = re.search(r"^Moves; (.+)$", line)
        if m:
            toks += m.group(1).split()
            last = None
            continue
        m = PV_TOK.search(line)
        if m:
            t = m.group(1)
            if t != last:
                toks.append(t)
            last = t
        m = SCORE.search(line)
        if m:
            evs.append(int(m.group(1)))
    b = _chess.Board()
    legal = []
    for t in toks:
        try:
            b.push(_chess.Move.from_uci(t))
            legal.append(t)
        except Exception:
            break
    return legal, (evs[-1] if evs else None)


def standings_and_h2h(pairings):
    pts = {p: 0.0 for p in ALL_PLAYERS}
    ng = {p: 0 for p in ALL_PLAYERS}
    hh = {a: {b: [0.0, 0] for b in ALL_PLAYERS} for a in ALL_PLAYERS}
    for pr in pairings:
        for g in pr["games"]:
            if g["result"] not in RESULT_CP:
                continue
            sc = RESULT_CP[g["result"]]
            w, b = g["white"], g["black"]
            if w in pts and b in pts:
                pts[w] += sc; ng[w] += 1
                pts[b] += 1 - sc; ng[b] += 1
                hh[w][b][0] += sc; hh[w][b][1] += 1
                hh[b][w][0] += 1 - sc; hh[b][w][1] += 1
    table = [{"p": p, "pts": round(pts[p], 1), "games": ng[p],
              "pct": round(100 * pts[p] / max(ng[p], 1), 1)}
             for p in sorted(ALL_PLAYERS, key=lambda q: (-pts[q], q))]
    return table, hh


class Analyzer:
    """Neutral SF17.1 analysis of one game, cached."""
    def __init__(self):
        self.p = None

    def _ensure(self):
        if self.p is None or self.p.poll() is not None:
            import subprocess
            self.p = subprocess.Popen([SF17], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
            self.p.stdin.write("uci\nisready\n"); self.p.stdin.flush()
            while "readyok" not in self.p.stdout.readline():
                pass
            self.p.stdin.write("setoption name Threads value 2\nsetoption name Hash value 256\nisready\n")
            self.p.stdin.flush()
            while "readyok" not in self.p.stdout.readline():
                pass

    def eval_fen(self, fen, depth):
        self._ensure()
        self.p.stdin.write(f"position fen {fen}\ngo depth {depth}\n"); self.p.stdin.flush()
        cp, bm = None, None
        for _ in range(4000):
            line = self.p.stdout.readline()
            if not line:
                break
            m = re.search(r"depth \d+ .*score cp (-?\d+).* pv (\S+)", line)
            if m:
                cp, bm = int(m.group(1)), m.group(2)
            if line.startswith("bestmove"):
                if line.split()[1] != "(none)":
                    bm = line.split()[1]
                break
        return cp, bm

    def analyze(self, pgn_text):
        game = _chess_pgn.read_game(__import__("io").StringIO(pgn_text))
        if game is None:
            return None
        board = game.board()
        fens = [board.fen()]
        moves = []
        for node in game.mainline():
            moves.append(node.move.uci())
            board.push(node.move)
            fens.append(board.fen())
        evals = []
        for fen in fens:
            stm_cp, bm = self.eval_fen(fen, AN_DEPTH)
            stm = -1 if " b " in fen else 1
            evals.append({"cp": stm_cp * stm if stm_cp is not None else None,
                          "best": bm})
        return {"depth": AN_DEPTH, "plies": len(moves), "moves": moves,
                "evals": evals}


ANALYZER = Analyzer()


def analyze_cached(pair, gidx):
    os.makedirs(f"{TDIR}/analysis", exist_ok=True)
    cache = f"{TDIR}/analysis/{pair}_{gidx}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    txt = open(f"{TDIR}/{pair}.pgn").read()
    pgns = ["[Event" + b for b in txt.split("[Event")[1:]]
    if gidx >= len(pgns):
        return {"error": "no such game"}
    data = ANALYZER.analyze(pgns[gidx])
    if data is None:
        return {"error": "parse failed"}
    json.dump(data, open(cache, "w"))
    return data


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>flychess tournament explorer</title>
<script src="/chess.js"></script>
<style>
:root{--bg:#161512;--panel:#262421;--panel2:#302e2b;--fg:#bababa;--br:#3d3a37;
      --acc:#56b04c;--gold:#d0a24a;--sq-l:#f0d9b5;--sq-d:#b58863}
*{box-sizing:border-box} body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);
 color:var(--fg);margin:0;padding:16px;font-size:14px}
h1{font-size:19px;margin:0 0 2px;color:#ddd} .sub{color:#8f8f8f;font-size:12px;margin-bottom:12px}
.bar{height:5px;background:var(--panel2);border-radius:3px;margin:6px 0 14px;max-width:860px}
.bar>div{height:100%;background:var(--acc);border-radius:3px}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.panel{background:var(--panel);border-radius:8px;padding:12px}
table{border-collapse:collapse;font-size:13px} td,th{padding:4px 9px;border-bottom:1px solid var(--br)}
th{color:#9f9f9f;font-weight:600;text-align:right} td{text-align:right}
td.l,th.l{text-align:left}
.ct td,.ct th{border:1px solid var(--br);min-width:44px;text-align:center}
.ct .tot{color:var(--gold)}
.sect{font-size:13px;color:#9f9f9f;margin:10px 0 6px;text-transform:uppercase;letter-spacing:.08em}
.pcard{background:var(--panel);border-radius:8px;padding:8px 10px;margin-bottom:8px;min-width:290px}
.pcard .ph{display:flex;justify-content:space-between;cursor:pointer}
.pcard .ph:hover{color:#fff}
.pcard.sel{outline:2px solid var(--acc)}
.sc{color:var(--gold)} .live-dot{color:var(--acc);font-weight:700}
.grow{display:flex;justify-content:space-between;padding:4px 6px;border-radius:5px;cursor:pointer;font-size:13px}
.grow:hover{background:var(--panel2)} .grow.sel{background:#3f4d3b}
.grow .res{font-weight:600} .r-w{color:#7fc67a}.r-l{color:#d07070}.r-d{color:#c9b458}
#viewer{background:var(--panel);border-radius:8px;padding:14px;min-width:520px}
#vhead{font-size:13px;color:#9f9f9f;margin-bottom:8px;min-height:17px}
#board{display:grid;grid-template-columns:repeat(8,52px);grid-auto-rows:52px;border-radius:4px;
 overflow:hidden;box-shadow:0 4px 14px rgba(0,0,0,.5)}
.sq{position:relative;display:flex;align-items:center;justify-content:center}
.sq img{width:48px;height:48px;-webkit-user-drag:none;user-select:none}
.sq.l{background:var(--sq-l)} .sq.d{background:var(--sq-d)}
.sq .co{position:absolute;font-size:9px;color:#888;font-weight:700}
.sq .co.f{bottom:0;right:2px} .sq .co.r{top:0;left:2px}
.sq.last::after{content:"";position:absolute;inset:0;background:#9bc70055}
#ctrl{margin:10px 0;display:flex;gap:5px;flex-wrap:wrap}
button{background:var(--panel2);border:1px solid var(--br);color:var(--fg);border-radius:5px;
 padding:5px 11px;cursor:pointer;font-size:14px} button:hover{background:#3d3a37}
button.on{background:#3f5d3a;border-color:var(--acc);color:#cfe8c9}
button:disabled{opacity:.4;cursor:default}
#evwrap{max-width:430px;margin-top:8px}
#evbar{height:16px;background:#404040;border-radius:3px;overflow:hidden;position:relative}
#evbar>div{height:100%;background:#e8e8e8;transition:width .5s}
#evlab{font-size:11px;color:#9f9f9f;margin-top:2px;min-height:14px}
#moves{max-height:280px;overflow-y:auto;background:var(--panel2);border-radius:6px;
 padding:4px 6px;max-width:430px}
.mrow{display:grid;grid-template-columns:30px 1fr 1fr;font-size:13.5px;line-height:2}
.mnum{color:#8f8f8f;text-align:right;padding-right:6px}
.mv{cursor:pointer;border-radius:4px;padding:0 6px;display:inline-block;min-width:40px}
.mv:hover{background:#4a4744} .mv.cur{background:#5a764f;color:#fff}
.mv.blun{color:#e08080;font-weight:700} .mv.mist{color:#d8b95e}
.mv .ev{color:#7f8f7f;font-size:10.5px;margin-left:4px}
#turnpts{max-width:430px;margin-top:6px;font-size:12.5px}
.tp{padding:4px 8px;border-radius:5px;background:var(--panel2);margin-top:4px;cursor:pointer}
.tp:hover{background:#3d3a37}
.leg{font-size:12px;color:#8f8f8f;margin-top:14px} .leg b{color:#c8b68a}
a{color:#7aa2d8}
</style></head><body>
<h1>flychess checkpoint-team round robin — FINAL</h1>
<div class="sub" id="sub">loading…</div>
<div class="bar"><div id="prog" style="width:0%"></div></div>
<div class="row">
 <div>
  <div class="panel"><div class="sect">standings</div><table id="st"></table></div>
  <div class="panel" style="margin-top:14px"><div class="sect">cross table <span style="text-transform:none;letter-spacing:0">(row vs col — row pts)</span></div><table class="ct" id="ct"></table></div>
 </div>
 <div id="plist"></div>
 <div id="viewer">
   <div id="vhead">select a pairing</div>
   <div class="row" style="gap:14px">
     <div>
       <div id="board"></div>
       <div id="evwrap"><div id="evbar"><div id="evfill" style="width:50%"></div></div><div id="evlab"></div></div>
     </div>
     <div>
       <div id="glist"></div>
       <div id="ctrl">
         <button onclick="jump(0)" title="start (Home)">|&#9664;</button>
         <button onclick="step(-1)" title="prev (&larr;)">&#9664;</button>
         <button onclick="step(1)" title="next (&rarr;)">&#9654;</button>
         <button onclick="jump(1e5)" title="end (End)">&#9654;|</button>
         <button id="flipb" onclick="FLIP=!FLIP;render()">flip</button>
         <button id="auto" onclick="autopl()">auto</button>
         <button id="anb" onclick="toggleAnalyze()" title="Stockfish 17.1 (neutral judge), depth 18 on every position — cached forever after first run">SF17 eval</button>
         <button id="pgnb" onclick="dlPGN()" title="download PGN">pgn</button>
       </div>
     </div>
   </div>
   <div id="moves">no game selected</div>
   <div id="turnpts"></div>
 </div>
</div>
<div class="leg"><b>players</b> — __LEGEND__<br>
analyze: neutral Stockfish 17.1, depth 18 per position (cached). per-move loss = how much the mover's eval dropped by their own move; yellow &ge;100cp, red &ge;250cp. turning points list the biggest losses.<br>
live game reconstruction: fastchess logs PVs per move (no per-move PGN flush exists); first PV token per search = played move, legality-checked.</div>
<script>
let CHESS=null,HIST=[],PLY=0,FLIP=false,AUTOT=null,CUR=null,LIVEP=null,PGNS=[],GMETA=[],GIDX=0,
    LIVEEV=null,AN=null,ANBUSY=false;
let VIEW={pair:null,game:-2};   // game: -2 nothing, -1 live, >=0 finished index
let SIG='';                      // pairing-set signature to avoid card churn
const GLY={k:"K",q:"Q",r:"R",b:"B",n:"N",p:"P"};
const $=id=>document.getElementById(id);
async function refresh(){
  const d = await (await fetch('/api/status')).json();
  $('sub').textContent = 'updated ' + d.time + '  ·  ' + d.total_games + '/126 games  ·  ' +
    d.pairings.filter(p=>p.active).length + ' playing now';
  $('prog').style.width = (100*d.total_games/168).toFixed(1)+'%';
  let h = '<tr><th class="l">#</th><th class="l">player</th><th>pts</th><th>games</th><th>%</th></tr>';
  d.table.forEach((v,i)=>{h+=`<tr><td class="l">${i+1}</td><td class="l">${v.p}</td><td class="sc">${v.pts}</td><td>${v.games}</td><td>${v.pct}</td></tr>`;});
  $('st').innerHTML=h;
  h='<tr><th class="l"></th>'; for(const c of d.players) h+=`<th>${c}</th>`; h+='<th>tot</th></tr>';
  for(const r of d.players){ let tot=0; h+=`<tr><th class="l">${r}</th>`;
    for(const c of d.players){ if(r===c){h+='<td style="color:#555">·</td>';continue;}
      const cell=d.h2h[r][c]; tot+=cell[0]; h+=`<td>${cell[1]?cell[0].toFixed(1):''}</td>`; }
    h+=`<td class="tot">${tot.toFixed(1)}</td></tr>`; }
  $('ct').innerHTML=h;
  const sig = d.pairings.map(x=>x.a+'_'+x.b+'_'+x.games.length+'_'+(x.active?1:0)).join('|');
  if (sig !== SIG){
    SIG = sig;
    let p='';
    for(const pr of d.pairings){
      const id=pr.a+'_vs_'+pr.b;
      p+=`<div class="pcard" id="pc_${id}"><div class="ph" onclick="show('${id}')">`+
         `<span>${pr.a} — ${pr.b}</span><span id="pcs_${id}"></span></div></div>`;
    }
    $('plist').innerHTML=p;
  }
  for(const pr of d.pairings){
    const id=pr.a+'_vs_'+pr.b, n=pr.games.length; let sc=0;
    for(const g of pr.games){ sc+=scOf(g,pr.a); }
    const el=$('pcs_'+id), card=$('pc_'+id);
    if(el) el.innerHTML=`<span class="sc">${sc.toFixed(1)}</span>/${n}${n>=6?' ✓':''}`+
      `${pr.active?' <span class="live-dot">● live</span>':''}`;
    if(card) card.classList.toggle('sel', CUR===id);
  }
  if (CUR && VIEW.pair===CUR && VIEW.game===-1) loadLive(false);
}
function scOf(g,a){ if(g.result==='1-0')return g.white===a?1:0; if(g.result==='0-1')return g.black===a?1:0;
  if(g.result==='1/2-1/2')return .5; return 0; }
const cls=r=>r==='1-0'||r==='0-1'?'r-w':r==='1/2-1/2'?'r-d':'r-l';
async function show(id){
  CUR=id; PLY=0; AN=null; $('turnpts').innerHTML='';
  VIEW={pair:id, game:-2};
  const d=await(await fetch('/api/status')).json();
  const pair=d.pairings.find(x=>x.a+'_vs_'+x.b===id);
  LIVEP=pair&&pair.active?id:null;
  await rebuildGlist();
  if(GMETA.length) playGame(0);
  else if(LIVEP) viewLive();
  refresh();
}
async function rebuildGlist(){
  const pd=await(await fetch('/api/pgn?p='+CUR)).json();
  PGNS=pd.pgns; GMETA=pd.games;
  let g='';
  if(LIVEP) g+=`<div class="grow" style="background:#2e3d2b" onclick="viewLive()"><span class="live-dot">● live game</span><span id="livemeta"></span></div>`;
  GMETA.forEach((gm,i)=>{g+=`<div class="grow" id="gr${i}" onclick="playGame(${i})"><span>${i+1}. ${gm.white}–${gm.black}</span><span class="res ${cls(gm.result)}">${gm.result==='*'?'…':gm.result}</span></div>`;});
  $('glist').innerHTML=g;
}
function viewLive(){ VIEW={pair:CUR, game:-1}; $('anb').disabled=true; AN=null; $('turnpts').innerHTML=''; loadLive(true); }
function markSel(i){ document.querySelectorAll('#glist .grow').forEach(e=>e.classList.remove('sel'));
  if(i>=0){const el=$('gr'+i); if(el)el.classList.add('sel');} }
function playGame(i){
  markSel(i); GIDX=i; VIEW={pair:CUR, game:i}; AN=null; $('turnpts').innerHTML='';
  $('evlab').textContent='no SF17 eval yet — click "SF17 eval" (first run ~1min, then cached)';
  CHESS=new Chess(); CHESS.load_pgn(PGNS[i],{sloppy:true});
  HIST=CHESS.history({verbose:true}); LIVEEV=null;
  $('anb').classList.remove('on'); $('anb').disabled=false;
  if(!HIST.length){ $('vhead').textContent='could not load game PGN'; }
  $('vhead').textContent=`${GMETA[i].white} (W) vs ${GMETA[i].black} (B) · ${GMETA[i].result} · ${HIST.length} ply`;
  jump(HIST.length);
}
async function loadLive(click){
  if (VIEW.pair!==CUR || VIEW.game!==-1) return;   // never stomp a chosen game
  if(click) markSel(-1);
  const d=await(await fetch('/api/live?p='+LIVEP)).json();
  if (VIEW.pair!==CUR || VIEW.game!==-1) return;
  CHESS=null; HIST=[]; PLY=0; LIVEEV=d.eval;
  const lm=$('livemeta');
  if(lm) lm.textContent=d.moves.length+' ply (recon)';
  $('vhead').textContent=LIVEP.replace('_vs_',' vs ')+' · LIVE — board unavailable mid-game '+
    '(fastchess logs are PV-streams; finished games render from PGN). '+
    'engine eval (side-to-move view): '+(LIVEEV===null?'n/a':(LIVEEV>0?'+':'')+LIVEEV+'cp');
  $('board').innerHTML='';
  $('moves').innerHTML='<div class="sect">live</div>game in progress — the finished games in this pairing are viewable now; this one appears the moment it ends.';
  $('evfill').style.width='50%'; $('evlab').textContent='live eval is side-to-move, not white-relative';
}
async function toggleAnalyze(){
  if(ANBUSY) return;
  if(AN){ AN=null; $('anb').classList.remove('on'); $('anb').textContent='SF17 eval'; $('turnpts').innerHTML=''; render(); return; }
  if(LIVEP||!GMETA.length) return;
  ANBUSY=true; $('anb').textContent='SF17 analyzing (≈1min)…';
  const d=await(await fetch(`/api/analyze?p=${CUR}&g=${GIDX}`)).json();
  ANBUSY=false; $('anb').textContent='SF17 eval ✓';
  if(d.error){ $('vhead').textContent='analysis failed: '+d.error; return; }
  AN=d; $('anb').classList.add('on');
  turnPoints(); render();
}
function winPct(cp){ return 100/(1+Math.exp(-cp/300)); }
function moveLoss(k){ // eval (mover persp) before move k vs after; positive = mover hurt self
  if(!AN) return 0;
  const before=AN.evals[k], after=AN.evals[k+1];
  if(before.cp==null||after.cp==null) return 0;
  const mover = k%2===0 ? 1 : -1;           // white moves on even ply
  return (before.cp*mover) - (after.cp*mover);
}
function turnPoints(){
  if(!AN) return;
  const items=[];
  for(let k=0;k<HIST.length;k++){
    const L=moveLoss(k);
    if(L>=150) items.push({k,L});
  }
  items.sort((a,b)=>b.L-a.L);
  let h='<div class="sect">turning points (mover self-harm)</div>';
  items.slice(0,6).forEach(it=>{
    const mv=HIST[it.k], who=it.k%2===0?'White':'Black';
    h+=`<div class="tp" onclick="jump(${it.k+1})"><b>${Math.floor(it.k/2)+1}${it.k%2?'…':''} ${mv.san}</b> — ${who} drops ${Math.round(it.L/100.*100)/100} pawns</div>`;
  });
  $('turnpts').innerHTML=h||'<div class="sect">turning points</div>none found';
}
function render(){
  if(!CHESS){$('board').innerHTML='';return;}
  const c=new Chess(); for(let k=0;k<PLY;k++) c.move(HIST[k]);
  const b=c.board(), last=PLY>0?HIST[PLY-1]:null, files='abcdefgh';
  let h='';
  for(let vr=0;vr<8;vr++){ const r=FLIP?vr:7-vr;
    for(let vf=0;vf<8;vf++){ const f=FLIP?7-vf:vf;
      const sq=b[7-r][f], dark=(r+f)%2===0, name=files[f]+(r+1);
      const isL=last&&(last.from===name||last.to===name);
      const co=(vf===7?`<span class="co f">${r+1}</span>`:'')+(vr===7?`<span class="co r">${files[f]}</span>`:'');
      const img=sq?`<img src="/pieces/${sq.color}${GLY[sq.type]}.svg" alt="">`:'';
      h+=`<div class="sq ${dark?'d':'l'} ${isL?'last':''}">${img}${co}</div>`;
    } }
  $('board').innerHTML=h;
  let m='', k=0;
  while(k<HIST.length){
    const n=Math.floor(k/2)+1;
    let wm = HIST[k]?HIST[k].san:'', wl=classFor(k), wv=evTag(k); k++;
    let bm = HIST[k]?HIST[k].san:'', bl=classFor(k), bv=evTag(k); if(HIST[k])k++;
    m+=`<div class="mrow"><span class="mnum">${n}.</span>`+
       `<span class="mv ${wl}" onclick="jump(${k-1})">${wm}${wv}</span>`+
       `<span class="mv ${bl}" onclick="jump(${k})">${bm}${bv}</span></div>`;
  }
  $('moves').innerHTML=m||'(no moves)';
  const cur=PYL_el(), box=$('moves');
  if(cur && box){
    const cTop = cur.offsetTop, cBot = cTop + cur.offsetHeight;
    if (cBot > box.scrollTop + box.clientHeight - 8) box.scrollTop = cBot - box.clientHeight + 8;
    else if (cTop < box.scrollTop + 8) box.scrollTop = cTop - 8;
  }
  let cp=LIVEEV;
  if(AN&&AN.evals[PLY]&&AN.evals[PLY].cp!=null) cp=AN.evals[PLY].cp;
  const pct=cp==null?50:winPct(cp);
  $('evfill').style.width=pct+'%';
  $('evlab').textContent = cp==null ? '' :
    (AN?'SF17 d'+AN.depth+': ':'engine: ')+(cp>0?'+':'')+cp+'cp'+(AN&&AN.evals[PLY]&&AN.evals[PLY].best?' · best '+(PLY<HIST.length?vsSan(AN.evals[PLY].best):''):'');
}
function PYL_el(){ const cells=document.querySelectorAll('#moves .mv.cur'); return cells[cells.length-1]; }
function classFor(k){
  let c = k===PLY?'cur':'';
  if(AN){ const L=moveLoss(k); if(L>=250) c+=' blun'; else if(L>=100) c+=' mist'; }
  return c;
}
function evTag(k){ if(!AN||!AN.evals[k+1]||AN.evals[k+1].cp==null) return '';
  const cp=AN.evals[k+1].cp; return `<span class="ev">${(cp>0?'+':'')+(cp/100).toFixed(1)}</span>`; }
function vsSan(uci){ const c=new Chess(); for(let k=0;k<PLY;k++)c.move(HIST[k]);
  const mv=c.move({from:uci.slice(0,2),to:uci.slice(2,4),promotion:uci[4]||'q'});
  return mv?mv.san:uci; }
function step(d){ jump(PLY+d); }
function jump(n){ PLY=Math.max(0,Math.min(n,HIST.length)); render(); }
function autopl(){
  if(AUTOT){clearInterval(AUTOT);AUTOT=null;$('auto').classList.remove('on');return;}
  $('auto').classList.add('on');
  AUTOT=setInterval(()=>{ if(PLY>=HIST.length){autopl();return;} step(1); },800);
}
function dlPGN(){
  if(!PGNS.length) return;
  const blob=new Blob([PGNS[GIDX]],{type:'application/x-chess-pgn'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download=CUR+'_'+(GIDX+1)+'.pgn'; a.click();
}
document.addEventListener('keydown',e=>{
  if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){
    e.preventDefault();
    if(e.key==='ArrowLeft')step(-1); else if(e.key==='ArrowRight')step(1);
    else if(e.key==='Home')jump(0); else jump(1e5);
  }});
refresh(); setInterval(refresh,6000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/status":
            prs = read_pairings()
            table, hh = standings_and_h2h(prs)
            self._send(json.dumps({
                "time": time.strftime("%H:%M:%S"),
                "total_games": sum(len(p["games"]) for p in prs),
                "table": table, "h2h": hh, "players": ALL_PLAYERS,
                "pairings": prs}).encode(), "application/json")
        elif u.path == "/api/pgn":
            p = parse_qs(u.query).get("p", [""])[0]
            f = os.path.join(TDIR, p + ".pgn")
            if os.path.isfile(f):
                txt = open(f).read()
                pgns = ["[Event" + b for b in txt.split("[Event")[1:]]
                games = next((x["games"] for x in read_pairings()
                              if x["a"] + "_vs_" + x["b"] == p), [])
            else:
                pgns, games = [], []
            self._send(json.dumps({"games": games, "pgns": pgns}).encode(),
                       "application/json")
        elif u.path == "/api/live":
            p = parse_qs(u.query).get("p", [""])[0]
            toks, ev = live_state(os.path.join(TDIR, p + ".log"))
            self._send(json.dumps({"moves": toks, "eval": ev}).encode(),
                       "application/json")
        elif u.path == "/api/analyze":
            q = parse_qs(u.query)
            p, g = q.get("p", [""])[0], int(q.get("g", ["0"])[0])
            try:
                self._send(json.dumps(analyze_cached(p, g)).encode(),
                           "application/json")
            except Exception as e:
                self._send(json.dumps({"error": str(e)}).encode(),
                           "application/json")
        elif u.path == "/chess.js":
            self._send(open(os.path.join(WEB, "chess.min.js"), "rb").read(),
                       "application/javascript")
        elif u.path.startswith("/pieces/"):
            f = os.path.join(WEB, "pieces", os.path.basename(u.path))
            if os.path.isfile(f):
                self._send(open(f, "rb").read(), "image/svg+xml")
            else:
                self._send(b"404", "text/plain", 404)
        else:
            page = PAGE.replace("__LEGEND__",
                                " · ".join(f"<b>{k}</b> {v}" for k, v in LEGEND))
            self._send(page.encode(), "text/html")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"serving {TDIR} on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
