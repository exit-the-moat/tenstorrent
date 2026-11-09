"""Minimal local chat with the model server of this package (OpenAI-compatible /v1/chat/completions). Standard library only.

    python3 chat/chat.py --no-tunnel                # the model server runs on this machine, port 8087
    python3 chat/chat.py --no-tunnel --tunnel-port 8093
    python3 chat/chat.py --ssh-host myserver        # the model server runs on another machine: opens an ssh tunnel
    then open http://127.0.0.1:8790

The model server listens on 127.0.0.1 of its machine only. From another machine, chat.py opens an SSH tunnel
(ssh -N -L) and proxies the browser's requests through it. Shows time to first token and decode tok/s separately for
every reply (decode = tokens after the first / time after the first). Nothing leaves this machine except the requests
to the model server.
"""
from __future__ import annotations

import argparse
import http.client
import json
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Local chat</title>
<style>
:root{--bg:#012243;--panel:#06315a;--text:#F8F6E9;--muted:#9fb3c8;--teal:#12D5D1;--amber:#FEB41F;--line:#0b4677}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 "IBM Plex Sans",system-ui,sans-serif;
display:flex;flex-direction:column;height:100vh}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header b{color:var(--teal)}header label{color:var(--muted);font-size:14px}
select,input[type=number]{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:3px 6px}
#log{flex:1;overflow-y:auto;padding:16px;max-width:900px;width:100%;margin:0 auto}
.m{margin:0 0 14px;padding:10px 14px;border-radius:10px;white-space:pre-wrap;word-wrap:break-word}
.u{background:var(--panel);border-left:3px solid var(--amber)}.a{border-left:3px solid var(--teal)}
.think{color:var(--muted);font-size:14px;border-left:2px solid var(--line);padding-left:8px;margin-bottom:6px}
.stat{color:var(--muted);font-size:13px;margin-top:6px;font-family:"IBM Plex Mono",monospace}
footer{border-top:1px solid var(--line);padding:10px 16px}
form{display:flex;gap:8px;max-width:900px;margin:0 auto}
textarea{flex:1;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px;
font:inherit;resize:vertical;min-height:52px}
button{background:var(--teal);color:var(--bg);border:0;border-radius:8px;padding:0 18px;font-weight:600;cursor:pointer}
button.sec{background:transparent;color:var(--muted);border:1px solid var(--line)}
</style></head><body>
<header><b>Local chat</b><span id="model" style="color:var(--muted);font-size:14px">…</span>
<label><input type="checkbox" id="think"> thinking</label>
<label>max tokens <input type="number" id="maxt" value="1024" min="16" max="16384" step="16" style="width:80px"></label>
<button class="sec" id="clear" type="button">new chat</button></header>
<div id="log"></div>
<footer><form id="f"><textarea id="q" placeholder="Message (Enter sends, Shift+Enter = new line)"></textarea>
<button id="send">Send</button><button class="sec" id="stop" type="button" disabled>Stop</button></form></footer>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),send=document.getElementById('send'),
stopb=document.getElementById('stop');let msgs=[],ctrl=null;
fetch('/api/models').then(r=>r.json()).then(j=>{document.getElementById('model').textContent=(j.data&&j.data[0]&&j.data[0].id)||'?'})
 .catch(()=>{document.getElementById('model').textContent='server not reachable'});
function add(cls){const d=document.createElement('div');d.className='m '+cls;log.appendChild(d);return d}
function show(div,text){const i=text.indexOf('<think>'),j=text.indexOf('</think>');div.innerHTML='';
 if(i>=0){const t=document.createElement('div');t.className='think';t.textContent=text.slice(i+7,j>=0?j:undefined).trim();div.appendChild(t);
  text=j>=0?text.slice(j+8):''}else if(j>=0){const t=document.createElement('div');t.className='think';t.textContent=text.slice(0,j).trim();
  div.appendChild(t);text=text.slice(j+8)}
 div.appendChild(document.createTextNode(text.replace(/^\s+/,'')))}
document.getElementById('clear').onclick=()=>{msgs=[];log.innerHTML=''};
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit()}});
stopb.onclick=()=>ctrl&&ctrl.abort();
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const text=q.value.trim();if(!text||ctrl)return;q.value='';
 add('u').textContent=text;msgs.push({role:'user',content:text});const div=add('a');div.textContent='…';
 send.disabled=true;stopb.disabled=false;ctrl=new AbortController();
 const body={messages:msgs,stream:true,stream_options:{include_usage:true},max_tokens:+document.getElementById('maxt').value,
  temperature:0,chat_template_kwargs:{enable_thinking:document.getElementById('think').checked}};
 let out='',t0=performance.now(),t1=null,tEnd=null,ntok=null,nchunk=0;
 try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:ctrl.signal});
  if(!r.ok)throw new Error('HTTP '+r.status+': '+await r.text());
  const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
  for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let k;
   while((k=buf.indexOf('\n'))>=0){const line=buf.slice(0,k).trim();buf=buf.slice(k+1);
    if(!line.startsWith('data:'))continue;const d=line.slice(5).trim();if(d==='[DONE]')continue;
    let j;try{j=JSON.parse(d)}catch(_){continue}
    if(j.usage&&j.usage.completion_tokens)ntok=j.usage.completion_tokens;
    const c=j.choices&&j.choices[0]&&j.choices[0].delta&&(j.choices[0].delta.content||j.choices[0].delta.reasoning_content);
    if(c){if(t1===null)t1=performance.now();tEnd=performance.now();nchunk++;out+=c;show(div,out);log.scrollTop=log.scrollHeight}}}
 }catch(err){if(err.name!=='AbortError'){out+=(out?'\n':'')+'[error] '+err.message;show(div,out)}}
 msgs.push({role:'assistant',content:out.replace(/^[\s\S]*?<\/think>/,'').trim()});
 const st=document.createElement('div');st.className='stat';
 if(t1!==null){const ttft=(t1-t0)/1000,dt=(tEnd-t1)/1000;
  st.textContent=ntok?`first token ${ttft.toFixed(2)} s · ${ntok} tokens · decode ${dt>0?((ntok-1)/dt).toFixed(1):'–'} tok/s`
   :`first token ${ttft.toFixed(2)} s · ${nchunk} chunks (no token count from the server)`}
 else st.textContent='no output';div.appendChild(st);
 ctrl=null;send.disabled=false;stopb.disabled=true;q.focus()};
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssh-host", default="", help="ssh host of the model server (required unless --no-tunnel)")
    ap.add_argument("--remote-port", type=int, default=8087, help="model server port on the remote machine's 127.0.0.1")
    ap.add_argument("--tunnel-port", type=int, default=8087, help="local port of the model server (--no-tunnel) or local end of the ssh tunnel")
    ap.add_argument("--port", type=int, default=8790, help="local web page port")
    ap.add_argument("--no-tunnel", action="store_true", help="talk to 127.0.0.1:--tunnel-port directly (server on this machine)")
    a = ap.parse_args()
    if not a.no_tunnel and not a.ssh_host:
        ap.error("give --ssh-host HOST, or --no-tunnel if the model server runs on this machine")

    tunnel = None
    if not a.no_tunnel:
        tunnel = subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                                   "-L", f"127.0.0.1:{a.tunnel_port}:127.0.0.1:{a.remote_port}", a.ssh_host])
        for _ in range(50):  # wait until the tunnel accepts connections
            if tunnel.poll() is not None:
                sys.exit("ssh tunnel failed (is the port in use, or the ssh host not reachable?)")
            try:
                socket.create_connection(("127.0.0.1", a.tunnel_port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.2)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _upstream(self, method, path, body=None):
            c = http.client.HTTPConnection("127.0.0.1", a.tunnel_port, timeout=600)
            c.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
            return c, c.getresponse()

        def _fail(self, e):
            msg = f"model server not reachable on port {a.remote_port}: {e}".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

        def do_GET(self):
            if self.path == "/api/models":
                try:
                    c, r = self._upstream("GET", "/v1/models")
                    data = r.read()
                    c.close()
                except OSError as e:
                    return self._fail(e)
                self.send_response(r.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            data = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/api/chat":
                self.send_error(404)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                c, r = self._upstream("POST", "/v1/chat/completions", body)
            except OSError as e:
                return self._fail(e)
            self.send_response(r.status)
            self.send_header("Content-Type", r.getheader("Content-Type", "text/event-stream"))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:  # pass the stream through as it arrives
                while True:
                    chunk = r.read1(65536) if hasattr(r, "read1") else r.read(1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser pressed Stop: closing the upstream connection ends the request
            finally:
                c.close()

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"chat: http://127.0.0.1:{a.port}  (model server on local port {a.tunnel_port}); Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel:
            tunnel.terminate()


if __name__ == "__main__":
    main()
