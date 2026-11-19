// Room switcher + "New room" (local studio only): upload → pipeline stages → open the room.
const STAGES = [['upload', 'Upload'], ['world', 'World'], ['layout', 'Layout'], ['objects', 'Lux3D'], ['joints', 'Joints']] as const;

export async function setupRooms(current: string) {
  const btn = document.getElementById('room-name') as HTMLButtonElement;
  const menu = document.getElementById('rooms')!;
  btn.textContent = current;
  let rooms: string[] = [];
  try { rooms = (await (await fetch('/rooms/index.json', { cache: 'no-store' })).json()).rooms; } catch {}
  const studio = await fetch('/api/studio').then((r) => r.json()).then((j) => j.studio === true, () => false);
  menu.innerHTML = rooms.map((r) => `<a href="/?room=${r}"${r === current ? ' aria-current="page"' : ''}>${r}</a>`).join('') +
    (studio ? '<button class="new" id="new-room">New room</button>' : '');
  btn.addEventListener('click', () => { menu.hidden = !menu.hidden; });
  document.addEventListener('pointerdown', (e) => { if (!menu.contains(e.target as Node) && e.target !== btn) menu.hidden = true; });
  const file = document.getElementById('file') as HTMLInputElement;
  document.getElementById('new-room')?.addEventListener('click', () => { menu.hidden = true; file.click(); });
  file.addEventListener('change', () => { if (file.files?.[0]) build(file.files[0]); });
}

async function build(f: File) {
  const name = f.name.replace(/\.[^.]+$/, '');
  const panel = document.getElementById('building')!;
  const list = document.getElementById('b-stages')!;
  const line = document.getElementById('b-line')!;
  document.getElementById('b-name')!.textContent = name;
  list.innerHTML = STAGES.map(([k, label]) => `<li data-k="${k}">${label}<span>—</span></li>`).join('');
  panel.hidden = false;
  const t0 = performance.now();
  const started: Record<string, number> = { upload: t0 };
  const mark = (k: string) => {
    const idx = STAGES.findIndex(([s]) => s === k);
    list.querySelectorAll('li').forEach((li, i) => {
      li.classList.toggle('done', i < idx || k === 'done');
      li.classList.toggle('on', i === idx);
    });
    if (!started[k]) started[k] = performance.now();
  };
  const tick = setInterval(() => {
    list.querySelectorAll('li').forEach((li) => {
      const k = (li as HTMLElement).dataset.k!;
      if (started[k]) (li.querySelector('span') as HTMLElement).textContent = fmt((performance.now() - started[k]) / 1000);
    });
  }, 1000);
  mark('upload');
  const res = await fetch(`/api/build?name=${encodeURIComponent(name)}&file=${encodeURIComponent(f.name)}`, { method: 'POST', body: f });
  const fail = (msg: string) => { line.textContent = msg; clearInterval(tick); panel.addEventListener('click', () => { panel.hidden = true; }, { once: true }); };
  if (!res.ok) { fail(await res.text()); return; }
  const { id } = await res.json();
  const es = new EventSource(`/api/build/${id}/events`);
  es.onmessage = (ev) => {
    const e = JSON.parse(ev.data);
    if (e.stage) mark(e.stage);
    if (e.line) line.textContent = e.line.replace(/^\d\d:\d\d:\d\d\s*/, '');
    if (e.done) {
      es.close(); if (e.ok) clearInterval(tick);
      if (e.ok) location.href = `/?room=${e.id}`; else fail('failed: ' + line.textContent);
    }
  };
}
const fmt = (s: number) => (s < 90 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);
