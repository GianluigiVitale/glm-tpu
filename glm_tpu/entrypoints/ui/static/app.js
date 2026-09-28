/* Layout/palette adapted from the owner's as-pt interface; see docs/UI.md. */
'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const uid = () => crypto.randomUUID().replaceAll('-', '');
let state = {chats:[], jobs:[]}, current = localStorage.getItem('glm-current'), busy = false, loaded = false, lastRender = '', pending = null;
const drafts = new Map();
document.documentElement.dataset.theme = localStorage.getItem('glm-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

function markdown(text) {
  // Escape FIRST. No raw HTML, remote images, links, or executable attributes.
  const pieces = text.split(/(```[\s\S]*?(?:```|$))/g);
  return pieces.map(part => {
    if (part.startsWith('```')) return '<pre><code>' + esc(part.replace(/^```[^\n]*\n?/, '').replace(/```$/, '')) + '</code></pre>';
    return esc(part).split(/\n\s*\n/).map(block => {
      const inline = s => s.replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      if (/^#{1,3} /.test(block)) return block.split('\n').map(line => line.replace(/^(#{1,3}) (.*)$/, (_,h,t) => `<h${h.length}>${inline(t)}</h${h.length}>`)).join('<br>');
      if (block.split('\n').every(l => /^[-*] /.test(l))) return '<ul>' + block.split('\n').map(l => '<li>' + inline(l.slice(2)) + '</li>').join('') + '</ul>';
      if (block.split('\n').every(l => /^\d+\. /.test(l))) return '<ol>' + block.split('\n').map(l => '<li>' + inline(l.replace(/^\d+\. /,'')) + '</li>').join('') + '</ol>';
      return '<p>' + inline(block).replaceAll('\n','<br>') + '</p>';
    }).join('');
  }).join('');
}
async function api(data) {
  const response = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json','X-GLM-UI':'1'}, body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}
function toast(text) { $('toast').textContent=text; $('toast').classList.add('show'); setTimeout(()=>$('toast').classList.remove('show'),3500); }
function select(id) {
  drafts.set(current,$('q').value); current=id; localStorage.setItem('glm-current',id || '');
  $('q').value=drafts.get(id)||''; lastRender=''; render();
  $('sidebar').classList.remove('open'); $('hamb').setAttribute('aria-expanded','false');
}
async function fresh() {
  if(busy)return;busy=true;render();
  try {const r=await api({action:'create'}); await refresh(); select(r.id);}
  catch(e){toast(e.message);} finally{busy=false;render();$('q').focus();}
}
function welcome() {
  return `<div class="welcome"><div class="big">G</div><div class="kicker">Your thinking companion</div>
    <h2>What’s on your mind?</h2><p>A question, a first draft, a problem to work through.<br>Start a conversation with GLM-5.3.</p>
    <div class="examples">
      <button data-example="Explain a complex idea using an everyday example. Start with how a transformer pays attention."><b>Understand something</b>Make a complex idea feel simple</button>
      <button data-example="Help me write a clear introduction for a research project. Ask me about the problem and audience first."><b>Find the right words</b>Give a research idea a clear introduction</button>
      <button data-example="A shop sells notebooks for 4 euros each. I buy 3 and pay with 20 euros. How much change should I receive? Explain your calculation."><b>Work it out</b>Take a problem one step at a time</button>
      <button data-example="Help me review a piece of Python code for correctness and clarity. Ask me to paste the code first."><b>Build something</b>Think through code together</button>
    </div><p class="sub">Saved conversations · Maximum thinking effort</p></div>`;
}
function render() {
  if (loaded && current && !state.chats.some(c=>c.id===current)) current=null;
  const chat=state.chats.find(c=>c.id===current), jobs=new Map(state.jobs.map(j=>[j.id,j]));
  $('topTitle').textContent=chat?.title || 'New conversation';
  const active=state.jobs.filter(j=>['queued','generating'].includes(j.status));
  $('statusText').textContent=state.error ? 'Needs attention' : active.length ? `${active.length} active${active.length>1?' / queued':''}` : 'Model ready';
  $('status').classList.toggle('off',!!state.error); $('banner').textContent=state.error || '';
  if(Number.isInteger(state.capacity) && state.capacity>0)$('contextNote').textContent=`${Math.round(state.capacity/1024)}K context, including conversation and thinking. Chats take turns. Check important answers.`;
  $('convlist').innerHTML='<div class="lbl">Conversations</div>'+[...state.chats].reverse().map(c=>
    `<div class="conv ${c.id===current?'active':''}"><button class="ctitle" data-chat="${c.id}" title="${esc(c.title)}">${esc(c.title)}</button>
    <div class="acts"><button data-rename="${c.id}" aria-label="Rename ${esc(c.title)}" title="Rename">✎</button><button data-delete="${c.id}" aria-label="Delete ${esc(c.title)}" title="Delete">×</button></div></div>`).join('');
  const key=JSON.stringify([current,chat?.messages,state.jobs.filter(j=>j.chat===current)]);
  if(key!==lastRender) {
    const sc=$('scroll'), bottom=sc.scrollHeight-sc.scrollTop-sc.clientHeight<90;
    const opened=new Set([...document.querySelectorAll('details[open]')].map(d=>d.dataset.job));
    $('scroll').classList.toggle('empty',!chat?.messages.length);
    $('thread').innerHTML=!chat?.messages.length?welcome():chat.messages.map(m=> {
      if(m.role==='user')return `<article class="msg user"><div class="av">Y</div><div class="body"><div class="role">You</div><div class="txt">${esc(m.content)}</div></div></article>`;
      const j=jobs.get(m.job), waiting=['queued','generating'].includes(j.status);
      const thinking=j.thinking?`<details class="reasoning" data-job="${j.id}" ${opened.has(j.id)?'open':''}><summary>${waiting&&!j.answer?'Thinking…':'Thinking'} · ${j.output_tokens.toLocaleString()} total output tokens</summary><div class="thought">${esc(j.thinking)}</div></details>`:'';
      const status=j.status==='queued'?'Waiting for its turn':j.status==='generating'?(j.answer?'Writing…':'Thinking…'):j.status==='incomplete'?'Incomplete · '+esc(j.stop_cause):'';
      const meta=j.status==='complete'?`${j.output_tokens.toLocaleString()} output tokens · ${j.decode_tps.toFixed(2)} tok/s · ${j.prompt_tokens.toLocaleString()} input tokens`:status;
      return `<article class="msg bot"><div class="av">G</div><div class="body"><div class="role">GLM-5.3</div>${thinking}<div class="prose">${markdown(j.answer)}</div>
        ${waiting&&!j.answer?'<span class="typing" aria-hidden="true"><i></i><i></i><i></i></span>':''}
        <div class="mrow">${j.answer?`<button class="mact" data-copy="${j.id}">Copy answer</button>`:''}<span class="meta ${j.status==='incomplete'?'error':''}">${meta}</span></div></div></article>`;
    }).join('');
    if(bottom || !lastRender)sc.scrollTop=sc.scrollHeight; lastRender=key;
  }
  const blocked=chat?.messages.length && chat.messages.at(-1).status!=='complete';
  $('send').disabled=busy || !!blocked || !!state.error || !$('q').value.trim();
  $('q').disabled=busy; $('newBtn').disabled=busy;
  $('q').placeholder=blocked?'Start another conversation while this one finishes…':'Ask anything, or pick up where you left off…';
}
async function refresh() {
  const r=await fetch('/api/state'); if(!r.ok)throw new Error('Cannot reach the local chat service.');
  state=await r.json(); loaded=true; render();
}
async function send(event) {
  event.preventDefault(); const text=$('q').value.trim(); if(busy || !text || $('send').disabled)return;
  busy=true; render();
  try {
    if(!current) {const c=await api({action:'create'});current=c.id;localStorage.setItem('glm-current',current);}
    // A network retry of the same text uses the same identity, never a duplicate answer.
    if(!pending || pending.chat!==current || pending.text!==text)pending={action:'send',chat:current,text,id:uid()};
    await api(pending); pending=null; $('q').value=''; drafts.delete(current); $('q').style.height='auto';
    await refresh();
  } catch(e){toast(e.message);} finally{busy=false;render();}
}
$('composer').addEventListener('submit',send);
$('q').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();send(e);}});
$('q').addEventListener('input',()=>{$('q').style.height='auto';$('q').style.height=Math.min(200,$('q').scrollHeight)+'px';render();});
$('newBtn').addEventListener('click',fresh);
$('themeBtn').addEventListener('click',()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=theme;localStorage.setItem('glm-theme',theme);});
$('hamb').addEventListener('click',()=>{$('hamb').setAttribute('aria-expanded',String($('sidebar').classList.toggle('open')));});
$('thread').addEventListener('click',async e=>{
  const example=e.target.closest('[data-example]');if(example){$('q').value=example.dataset.example;$('q').focus();render();}
  const copy=e.target.closest('[data-copy]');if(copy){try{await navigator.clipboard.writeText(state.jobs.find(j=>j.id===copy.dataset.copy).answer);toast('Answer copied');}catch{toast('Copy is unavailable in this browser.');}}
});
$('convlist').addEventListener('click',async e=>{
  const c=e.target.closest('[data-chat]');if(c)return select(c.dataset.chat);
  const r=e.target.closest('[data-rename]'), d=e.target.closest('[data-delete]');
  try {
    if(r){const chat=state.chats.find(c=>c.id===r.dataset.rename),title=prompt('Conversation name',chat.title);if(title)await api({action:'rename',chat:chat.id,title});}
    if(d&&confirm('Delete this conversation from your workspace?'))await api({action:'delete',chat:d.dataset.delete});
    await refresh();
  }catch(e){toast(e.message);}
});
async function poll(){try{await refresh();}catch(e){$('statusText').textContent='Disconnected';$('status').classList.add('off');$('banner').textContent=e.message;$('send').disabled=true;}finally{setTimeout(poll,1500);}}
render();poll();
