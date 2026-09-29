// radio_tomo viewer: slice sweeping, contrast, processed PCA-RGB overlay, C-RADIO embedding jobs,
// torch-slicepick score plot. Talks to webapp/server.py.

interface TomoInfo {
  path: string;
  name: string;
  shape: [number, number, number];
  voxel: number;
  embedded: number[];
  pick: number | null;
}

interface Scores {
  z: number[];
  score: number[];
  base: number[];
  "entropy H": number[];
  "Laplacian var L (norm.)": number[];
  "edge density E (norm.)": number[];
  slab_z: number[];
  slab_score: number[];
  pick: number;
}

interface Job {
  state: "idle" | "running" | "done" | "error";
  path: string | null;
  done: number;
  total: number;
  message: string;
}

interface Overlay {
  img: HTMLImageElement;
  how: string;
}

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const SVGNS = "http://www.w3.org/2000/svg";

let tomo: TomoInfo | null = null;
let z = 0;
let selected = new Set<number>();
let done = new Set<number>();
let scores: Scores | null = null;
const slices = new Map<number, HTMLImageElement>(); // raw 8-bit slices (LRU)
const overlays = new Map<number, Overlay | null>();
let view = { scale: 1, tx: 0, ty: 0 };
let lut = new Uint8ClampedArray(256);
let raf = 0;

// ---------------------------------------------------------------- data
async function getJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function q(path: string, extra = ""): string {
  return `path=${encodeURIComponent(path)}${extra}`;
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(url));
    img.src = url;
  });
}

async function getSlice(k: number): Promise<HTMLImageElement> {
  const hit = slices.get(k);
  if (hit) {
    slices.delete(k);
    slices.set(k, hit); // refresh LRU order
    return hit;
  }
  const img = await loadImage(`/api/slice?${q(tomo!.path, `&z=${k}`)}`);
  slices.set(k, img);
  while (slices.size > 96) slices.delete(slices.keys().next().value!);
  return img;
}

async function getOverlay(k: number): Promise<Overlay | null> {
  if (overlays.has(k)) return overlays.get(k)!;
  const r = await fetch(`/api/overlay?${q(tomo!.path, `&z=${k}`)}`);
  if (!r.ok) {
    overlays.set(k, null);
    return null;
  }
  const how = r.headers.get("X-Overlay") ?? "";
  const img = await loadImage(URL.createObjectURL(await r.blob()));
  const o = { img, how };
  overlays.set(k, o);
  return o;
}

function prefetch(): void {
  if (!tomo) return;
  for (const d of [1, -1, 2, -2, 3, -3]) {
    const k = z + d;
    if (k >= 0 && k < tomo.shape[0] && !slices.has(k)) getSlice(k).catch(() => {});
  }
}

// ---------------------------------------------------------------- drawing
const canvas = $<HTMLCanvasElement>("canvas");
const ctx = canvas.getContext("2d")!;
const work = document.createElement("canvas"); // current slice after the contrast LUT
const wctx = work.getContext("2d", { willReadFrequently: true })!;
let workZ = -1;
let workKey = "";

function updateLut(): void {
  const lo = +$<HTMLInputElement>("cmin").value;
  const hi = Math.max(lo + 1, +$<HTMLInputElement>("cmax").value);
  for (let i = 0; i < 256; i++) lut[i] = Math.round(Math.min(255, Math.max(0, ((i - lo) / (hi - lo)) * 255)));
  workKey = "";
  requestDraw();
}

function applyContrast(img: HTMLImageElement): void {
  const k = `${z}:${$<HTMLInputElement>("cmin").value}:${$<HTMLInputElement>("cmax").value}`;
  if (workZ === z && workKey === k) return;
  work.width = img.naturalWidth;
  work.height = img.naturalHeight;
  wctx.drawImage(img, 0, 0);
  const d = wctx.getImageData(0, 0, work.width, work.height);
  const p = d.data;
  for (let i = 0; i < p.length; i += 4) {
    const v = lut[p[i]];
    p[i] = p[i + 1] = p[i + 2] = v;
  }
  wctx.putImageData(d, 0, 0);
  workZ = z;
  workKey = k;
}

function fit(): void {
  if (!tomo) return;
  const [, h, w] = tomo.shape;
  const r = canvas.getBoundingClientRect();
  view.scale = Math.min(r.width / w, r.height / h) * 0.96;
  view.tx = (r.width - w * view.scale) / 2;
  view.ty = (r.height - h * view.scale) / 2;
  requestDraw();
}

function requestDraw(): void {
  if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
}

async function draw(): Promise<void> {
  if (!tomo) return;
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  const want = z;
  const img = await getSlice(want);
  if (want !== z) return; // a newer slice was requested meanwhile
  applyContrast(img);
  const [, h, w] = tomo.shape;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--viewer") || "#111";
  ctx.fillRect(0, 0, r.width, r.height);
  ctx.setTransform(dpr * view.scale, 0, 0, dpr * view.scale, dpr * view.tx, dpr * view.ty);
  ctx.imageSmoothingEnabled = view.scale < 2;
  ctx.drawImage(work, 0, 0, w, h);

  let ovText = "";
  if ($<HTMLInputElement>("ovshow").checked && done.size) {
    const ov = await getOverlay(want);
    if (want !== z) return;
    if (ov) {
      ctx.imageSmoothingEnabled = false; // one colour block per 16-px token
      ctx.globalAlpha = +$<HTMLInputElement>("ovalpha").value / 100;
      ctx.drawImage(ov.img, 0, 0, w, h);
      ctx.globalAlpha = 1;
      ovText = ov.how;
    }
  }
  $("ovtag").textContent = ovText ? `overlay: ${ovText}` : done.size ? "no overlay for this slice" : "";
  $("hud").textContent = `z ${z} / ${tomo.shape[0] - 1}   ${(view.scale * 100).toFixed(0)}%`;
  prefetch();
}

// ---------------------------------------------------------------- slice navigation
function setZ(k: number): void {
  if (!tomo) return;
  z = Math.max(0, Math.min(tomo.shape[0] - 1, Math.round(k)));
  $<HTMLInputElement>("zslider").value = String(z);
  $<HTMLInputElement>("znum").value = String(z);
  const tag = done.has(z) ? "✓ embedded" : selected.has(z) ? "• selected" : "";
  $("ztag").textContent = tag;
  history.replaceState(null, "", `#${encodeURIComponent(tomo.path)}@${z}`);
  updateNowLines();
  highlightList();
  requestDraw();
}

// ---------------------------------------------------------------- embedding
function renderList(): void {
  const ul = $("zlist");
  ul.replaceChildren();
  for (const k of [...selected].sort((a, b) => a - b)) {
    const li = document.createElement("li");
    li.dataset.z = String(k);
    const label = document.createElement("span");
    label.textContent = `z ${k}`;
    const st = document.createElement("span");
    st.textContent = done.has(k) ? "✓ embedded" : "pending";
    st.className = done.has(k) ? "ok" : "muted";
    li.append(label, st);
    if (!done.has(k)) {
      const x = document.createElement("button");
      x.className = "x";
      x.textContent = "×";
      x.title = "remove";
      x.addEventListener("click", (ev) => { ev.stopPropagation(); selected.delete(k); renderList(); });
      li.append(x);
    }
    li.addEventListener("click", () => setZ(k));
    ul.append(li);
  }
  highlightList();
  const pending = [...selected].filter((k) => !done.has(k)).length;
  $<HTMLButtonElement>("embed").textContent = pending ? `Embed selected (${pending})` : "Embed selected";
}

function highlightList(): void {
  for (const li of document.querySelectorAll<HTMLLIElement>("#zlist li")) li.classList.toggle("cur", +li.dataset.z! === z);
}

async function startEmbed(): Promise<void> {
  if (!tomo) return;
  const zs = [...selected].filter((k) => !done.has(k));
  if (!zs.length) return;
  $<HTMLButtonElement>("embed").disabled = true;
  try {
    await getJSON<Job>("/api/embed", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: tomo.path, zs }),
    });
    pollJob();
  } catch (e) {
    $("jobmsg").textContent = String(e);
    $<HTMLButtonElement>("embed").disabled = false;
  }
}

async function pollJob(): Promise<void> {
  const j = await getJSON<Job>("/api/job");
  const prog = $<HTMLProgressElement>("prog");
  prog.max = Math.max(1, j.total);
  prog.value = j.done;
  $("jobmsg").textContent = j.total ? `${j.message} ${j.done}/${j.total}` : j.message;
  if (tomo && j.path === tomo.path) {
    const info = await getJSON<TomoInfo>(`/api/open?${q(tomo.path)}`);
    const before = done.size;
    done = new Set(info.embedded);
    if (done.size !== before) {
      overlays.clear(); // interpolation neighbours changed
      renderList();
      setZ(z);
    }
  }
  if (j.state === "running") setTimeout(pollJob, 600);
  else $<HTMLButtonElement>("embed").disabled = false;
}

// ---------------------------------------------------------------- score plot
interface Plot {
  svg: SVGSVGElement;
  x: (v: number) => number;
  now: SVGLineElement | null;
}
const plots: Plot[] = [];

function el(tag: string, attrs: Record<string, string | number>, parent?: Element): SVGElement {
  const e = document.createElementNS(SVGNS, tag) as SVGElement;
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.append(e);
  return e;
}

function drawPlot(svg: SVGSVGElement, series: { x: number[]; y: number[]; color: string; label: string; width?: number }[],
                  yLabel: string, pick: number | null): Plot {
  svg.replaceChildren();
  const r = svg.getBoundingClientRect();
  const W = Math.max(200, r.width), H = Math.max(80, r.height);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const m = { l: 34, r: 8, t: 6, b: 18 };
  const nz = tomo!.shape[0];
  const ymax = Math.max(1e-6, ...series.flatMap((s) => s.y));
  const x = (v: number) => m.l + (v / Math.max(1, nz - 1)) * (W - m.l - m.r);
  const y = (v: number) => H - m.b - (v / ymax) * (H - m.t - m.b);
  for (let i = 0; i <= 4; i++) {
    const v = (ymax * i) / 4;
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "grid" }, svg);
    const t = el("text", { x: m.l - 4, y: y(v) + 3, "text-anchor": "end" }, svg);
    t.textContent = v.toFixed(2);
  }
  const step = nz > 400 ? 100 : 50;
  for (let v = 0; v < nz; v += step) {
    const t = el("text", { x: x(v), y: H - 4, "text-anchor": "middle" }, svg);
    t.textContent = String(v);
  }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
  if (pick !== null) el("line", { x1: x(pick), x2: x(pick), y1: m.t, y2: H - m.b, class: "pick" }, svg);
  for (const s of series) {
    const pts = s.x.map((v, i) => `${x(v).toFixed(1)},${y(s.y[i]).toFixed(1)}`).join(" ");
    el("polyline", { points: pts, fill: "none", stroke: s.color, "stroke-width": s.width ?? 1 }, svg);
  }
  const lg = el("g", { class: "legend" }, svg);
  el("rect", { x: W - 156, y: m.t, width: 150, height: series.length * 12 + 6, rx: 4, class: "lgbox" }, lg);
  series.forEach((s, i) => {
    el("line", { x1: W - 150, x2: W - 136, y1: m.t + 8 + i * 12, y2: m.t + 8 + i * 12, stroke: s.color, "stroke-width": 2 }, lg);
    const t = el("text", { x: W - 132, y: m.t + 11 + i * 12 }, lg);
    t.textContent = s.label;
  });
  const now = el("line", { x1: x(z), x2: x(z), y1: m.t, y2: H - m.b, class: "now" }, svg) as SVGLineElement;
  svg.setAttribute("aria-label", yLabel);
  svg.onclick = (ev) => {
    const b = svg.getBoundingClientRect();
    const px = ((ev.clientX - b.left) / b.width) * W;
    setZ(((px - m.l) / (W - m.l - m.r)) * (nz - 1));
    $("viewer").focus();
  };
  return { svg, x, now };
}

function renderScores(): void {
  plots.length = 0;
  if (!scores || !tomo) return;
  const css = getComputedStyle(document.documentElement);
  const c = (v: string) => css.getPropertyValue(v).trim();
  plots.push(drawPlot($<SVGSVGElement & HTMLElement>("plotscore") as unknown as SVGSVGElement, [
    { x: scores.z, y: scores.base, color: c("--base"), label: "base (no centre prior)" },
    { x: scores.slab_z, y: scores.slab_score, color: c("--slab"), label: "per 4-slice slab" },
    { x: scores.z, y: scores.score, color: c("--score"), label: "score (per slice)", width: 1.4 },
  ], "score", scores.pick));
  plots.push(drawPlot($<SVGSVGElement & HTMLElement>("plotcomp") as unknown as SVGSVGElement, [
    { x: scores.z, y: scores["entropy H"], color: c("--c1"), label: "entropy H" },
    { x: scores.z, y: scores["Laplacian var L (norm.)"], color: c("--c2"), label: "Laplacian var L" },
    { x: scores.z, y: scores["edge density E (norm.)"], color: c("--c3"), label: "edge density E" },
  ], "component", null));
  const best = scores.z[scores.score.indexOf(Math.max(...scores.score))];
  $("stag").textContent = `— package pick z=${scores.pick}, best per-slice z=${best}; click to jump`;
}

function updateNowLines(): void {
  for (const p of plots) {
    if (!p.now) continue;
    p.now.setAttribute("x1", String(p.x(z)));
    p.now.setAttribute("x2", String(p.x(z)));
  }
}

// ---------------------------------------------------------------- open a tomogram
async function openTomo(path: string, startZ?: number): Promise<void> {
  tomo = await getJSON<TomoInfo>(`/api/open?${q(path)}`);
  slices.clear();
  overlays.clear();
  workZ = -1;
  done = new Set(tomo.embedded);
  selected = new Set(tomo.embedded);
  const [nz, h, w] = tomo.shape;
  $("info").textContent = `${w} × ${h} px · ${nz} slices · ${tomo.voxel.toFixed(2)} Å/px`;
  for (const id of ["zslider", "znum", "rfrom", "rto"]) $<HTMLInputElement>(id).max = String(nz - 1);
  $<HTMLInputElement>("rfrom").value = String(Math.floor(nz / 5));
  $<HTMLInputElement>("rto").value = String(nz - 1 - Math.floor(nz / 5));
  $<HTMLSelectElement>("tomo").value = path;
  renderList();
  fit();
  setZ(startZ ?? Math.floor(nz / 2));
  scores = null;
  renderScores();
  $("stag").textContent = "— computing…";
  scores = await getJSON<Scores>(`/api/scores?${q(path)}`);
  if (tomo.path === path) {
    renderScores();
    updateNowLines();
  }
  const j = await getJSON<Job>("/api/job");
  if (j.state === "running") pollJob();
}

// ---------------------------------------------------------------- events
function bind(): void {
  const viewer = $("viewer");
  $<HTMLInputElement>("zslider").addEventListener("input", (e) => setZ(+(e.target as HTMLInputElement).value));
  $<HTMLInputElement>("znum").addEventListener("change", (e) => setZ(+(e.target as HTMLInputElement).value));
  for (const id of ["cmin", "cmax"]) $(id).addEventListener("input", updateLut);
  for (const id of ["ovshow", "ovalpha"]) $(id).addEventListener("input", requestDraw);
  $("tomo").addEventListener("change", (e) => openTomo((e.target as HTMLSelectElement).value));
  $("addcur").addEventListener("click", () => { selected.add(z); renderList(); setZ(z); });
  $("addrange").addEventListener("click", () => {
    const a = +$<HTMLInputElement>("rfrom").value, b = +$<HTMLInputElement>("rto").value;
    const s = Math.max(1, +$<HTMLInputElement>("revery").value);
    for (let k = Math.min(a, b); k <= Math.max(a, b); k += s) selected.add(k);
    renderList();
    setZ(z);
  });
  $("clear").addEventListener("click", () => { selected = new Set(done); renderList(); setZ(z); });
  $("embed").addEventListener("click", startEmbed);

  window.addEventListener("keydown", (ev) => {
    if ((ev.target as HTMLElement).closest("input, select, textarea")) return;
    const big = ev.shiftKey ? 10 : 1;
    if (ev.key === "ArrowUp") setZ(z + big);
    else if (ev.key === "ArrowDown") setZ(z - big);
    else if (ev.key === "Home") setZ(0);
    else if (ev.key === "End") setZ(1e9);
    else if (ev.key === "a" || ev.key === "A") { selected.add(z); renderList(); setZ(z); }
    else if (ev.key === "o" || ev.key === "O") {
      const cb = $<HTMLInputElement>("ovshow");
      cb.checked = !cb.checked;
      requestDraw();
    } else return;
    ev.preventDefault();
  });

  viewer.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const b = canvas.getBoundingClientRect();
    const mx = ev.clientX - b.left, my = ev.clientY - b.top;
    const f = Math.exp(-ev.deltaY * 0.0015);
    view.tx = mx - (mx - view.tx) * f;
    view.ty = my - (my - view.ty) * f;
    view.scale *= f;
    requestDraw();
  }, { passive: false });
  let drag: { x: number; y: number } | null = null;
  viewer.addEventListener("pointerdown", (ev) => {
    drag = { x: ev.clientX, y: ev.clientY };
    viewer.setPointerCapture(ev.pointerId);
    viewer.classList.add("dragging");
  });
  viewer.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    view.tx += ev.clientX - drag.x;
    view.ty += ev.clientY - drag.y;
    drag = { x: ev.clientX, y: ev.clientY };
    requestDraw();
  });
  viewer.addEventListener("pointerup", () => { drag = null; viewer.classList.remove("dragging"); });
  viewer.addEventListener("dblclick", fit);
  new ResizeObserver(() => { requestDraw(); renderScores(); updateNowLines(); }).observe(document.body);
}

async function init(): Promise<void> {
  bind();
  updateLut();
  const list = await getJSON<string[]>("/api/tomograms");
  const sel = $<HTMLSelectElement>("tomo");
  for (const p of list) {
    const o = document.createElement("option");
    o.value = p;
    o.textContent = p;
    sel.append(o);
  }
  const m = decodeURIComponent(location.hash.slice(1)).match(/^(.*)@(\d+)$/);
  const start = m && list.includes(m[1]) ? m[1] : list.find((p) => p.includes("TS_030")) ?? list[0];
  if (start) await openTomo(start, m && m[1] === start ? +m[2] : undefined);
}

init().catch((e) => { $("info").textContent = `error: ${e}`; });
