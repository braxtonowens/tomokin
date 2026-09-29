"use strict";
// Slice finder + scribble segmentation on frozen C-RADIO features.
// Entropy dock: very smooth curve of the per-slice spectral entropy (outputs/entropy), ends cut, valleys between humps.
// Annotation: brush / eraser / lasso / magic wand (RADIO feature similarity) per slice, autosaved on the server;
// Every edit refits the U-Net on the painted pixels and predicts every slice (seg_interactive.py).
(() => {
    const CUT = 0.2, SSIG = 8, VMIN = 0.1;
    const PROJECT = new URLSearchParams(location.search).get("project") ?? ""; // pooled classifier over a tomogram list
    const pq = PROJECT ? `&project=${encodeURIComponent(PROJECT)}` : "";
    const PALETTE = ["#4f7cff", "#ff5c8a", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4", "#ef4444", "#84cc16", "#e879f9"];
    const UNASSIGNED = "#4f7cff"; // blue
    const $ = (id) => document.getElementById(id);
    const stage = $("stage"), view = $("view");
    const base = $("canvas"), annot = $("annot"), cursor = $("cursor");
    const denseC = $("dense"), nniC = $("nniobj");
    const bctx = base.getContext("2d"), actx = annot.getContext("2d"), cctx = cursor.getContext("2d");
    const predImg = $("pred"), unsImg = $("uns");
    const svg = $("plot");
    const NS = "http://www.w3.org/2000/svg";
    let info = null, z = 0, W = 1, H = 1;
    let curve = [], vals = [], lo = 0, hi = 1;
    let gray = null, pca = null, want = 0;
    let classes = [], cur = 1, tool = "brush", segmented = false, version = 0, unsure = [];
    let liveBusy = false, livePending = false, liveT = 0;
    // background prior (class 1 : rest), log scale 1..500 on a 0..100 slider; "auto" = ratio of painted pixels
    const PMAX = 500, toPrior = (v) => Math.exp((v / 100) * Math.log(PMAX)), toSlider = (p) => (100 * Math.log(p)) / Math.log(PMAX);
    let prior = 1, priorAuto = true;
    // "Clear guess": no prediction until the next edit (remembered across reloads / tomograms)
    let cleared = false, epoch = 0, explicitNext = false;
    let head = "unet", changedSinceFit = true; // scribbles edited since the last fit
    // (head fixed to the U-Net; the MLP / positive-only heads stay in seg_interactive.py)  // explicitNext: the next fit follows a stroke
    try {
        cleared = localStorage.getItem(`seg.cleared${PROJECT}`) === "1";
    }
    catch { /* storage unavailable */ }
    const setCleared = (v) => { cleared = v; try {
        localStorage.setItem(`seg.cleared${PROJECT}`, v ? "1" : "0");
    }
    catch { /* storage unavailable */ } };
    try {
        const s = JSON.parse(localStorage.getItem(`seg.prior${PROJECT}`) ?? "null");
        if (s) {
            prior = s.prior;
            priorAuto = s.auto;
        }
    }
    catch { /* storage unavailable */ }
    priorAuto = true; // the background prior is always automatic (no control)
    function showPrior() { }
    const labels = new Map(); // z -> H*W class ids (0 = none)
    const zClasses = new Map(); // z -> class ids painted on that slice (for the graph)
    function updateZ(zz) {
        const a = labels.get(zz), seen = new Set();
        if (a)
            for (let i = 0; i < a.length; i += 5)
                if (a[i])
                    seen.add(a[i]);
        if (seen.size)
            zClasses.set(zz, [...seen].sort());
        else
            zClasses.delete(zz);
    }
    const undo = [];
    let buf = null; // colored scribble layer for the shown slice
    let scale = 1, tx = 0, ty = 0;
    const on = (id) => $(id).getAttribute("aria-pressed") === "true";
    const setOn = (id, v) => $(id).setAttribute("aria-pressed", String(v));
    const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
    const load = (url) => new Promise((res, rej) => {
        const im = new Image();
        im.onload = () => res(im);
        im.onerror = rej;
        im.src = url;
    });
    function toast(msg, ms = 3500) {
        const t = $("toast");
        t.textContent = msg;
        t.hidden = false;
        clearTimeout(toast.h);
        toast.h = window.setTimeout(() => { t.hidden = true; }, ms);
    }
    // ------------------------------------------------------------ view transform
    function applyView() { view.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`; }
    function fit() {
        const r = stage.getBoundingClientRect();
        scale = Math.min(r.width / W, r.height / H) * 0.96;
        tx = (r.width - W * scale) / 2;
        ty = (r.height - H * scale) / 2;
        applyView();
    }
    function toImg(ev) {
        const r = stage.getBoundingClientRect();
        return [(ev.clientX - r.left - tx) / scale, (ev.clientY - r.top - ty) / scale];
    }
    const inGallery = (ev) => !!ev.target.closest(".gallery, .nnipanel, .classes, .tools");
    stage.addEventListener("wheel", (ev) => {
        if (inGallery(ev))
            return; // let the gallery scroll
        ev.preventDefault();
        const r = stage.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
        const f = Math.exp(-ev.deltaY * 0.0015), ns = Math.min(40, Math.max(0.1, scale * f));
        tx = mx - ((mx - tx) * ns) / scale;
        ty = my - ((my - ty) * ns) / scale;
        scale = ns;
        applyView();
        drawCursor(lastPt);
    }, { passive: false });
    stage.addEventListener("dblclick", (ev) => { if (inGallery(ev))
        return; if (tool === "pan")
        fit(); ev.preventDefault(); });
    // ------------------------------------------------------------ image layers
    function sizeLayers(w, h) {
        if (w === W && h === H && base.width === w)
            return;
        W = w;
        H = h;
        for (const c of [base, annot, cursor, denseC, nniC]) {
            c.width = W;
            c.height = H;
            c.style.width = `${W}px`;
            c.style.height = `${H}px`;
        }
        for (const im of [predImg, unsImg]) {
            im.style.width = `${W}px`;
            im.style.height = `${H}px`;
        }
        view.style.width = `${W}px`;
        view.style.height = `${H}px`;
        fit();
    }
    function drawBase() {
        if (!gray)
            return;
        bctx.globalAlpha = 1;
        bctx.imageSmoothingEnabled = true;
        bctx.drawImage(gray, 0, 0);
        const lo_ = +$("cmin").value, hi_ = Math.max(lo_ + 1, +$("cmax").value);
        if (lo_ > 0 || hi_ < 255) {
            const d = bctx.getImageData(0, 0, W, H), px = d.data;
            for (let i = 0; i < px.length; i += 4)
                px[i] = px[i + 1] = px[i + 2] = Math.max(0, Math.min(255, ((px[i] - lo_) * 255) / (hi_ - lo_)));
            bctx.putImageData(d, 0, 0);
        }
        if (on("tgOv") && pca) {
            bctx.globalAlpha = +$("ovalpha").value / 100;
            bctx.imageSmoothingEnabled = false;
            bctx.drawImage(pca, 0, 0, W, H);
            bctx.globalAlpha = 1;
        }
    }
    function lab() {
        let a = labels.get(z);
        if (!a) {
            a = new Uint8Array(W * H);
            labels.set(z, a);
        }
        return a;
    }
    function paintBuf(x0 = 0, y0 = 0, x1 = W, y1 = H) {
        if (!buf)
            buf = actx.createImageData(W, H);
        const a = labels.get(z), px = buf.data, cols = classes.map((c) => hex(c.color));
        for (let y = Math.max(0, y0); y < Math.min(H, y1); y++)
            for (let x = Math.max(0, x0); x < Math.min(W, x1); x++) {
                const i = y * W + x, k = a ? a[i] : 0, o = i * 4;
                if (k && cols[k - 1]) {
                    px[o] = cols[k - 1][0];
                    px[o + 1] = cols[k - 1][1];
                    px[o + 2] = cols[k - 1][2];
                    px[o + 3] = 255;
                }
                else
                    px[o + 3] = 0;
            }
        actx.putImageData(buf, 0, 0, Math.max(0, x0), Math.max(0, y0), Math.min(W, x1) - Math.max(0, x0), Math.min(H, y1) - Math.max(0, y0));
    }
    function refreshAnnot() {
        buf = null;
        annot.style.opacity = String(+$("annalpha").value / 100);
        paintBuf();
        counts();
    }
    async function refreshPred() {
        if (!info || !segmented) {
            predImg.hidden = true;
            unsImg.hidden = true;
            return;
        }
        const cols = classes.map((c) => c.color.slice(1)).join(",");
        const url = (mode) => `/api/seg/pred?run=${encodeURIComponent(info.run)}&z=${z}&colors=${cols}` +
            `&zavg=${$("zavg").checked ? 1 : 0}&mode=${mode}&v=${version}${pq}&prior=${prior.toFixed(4)}`;
        const id = want, v = version;
        const [p, u] = await Promise.all([on("tgPred") ? load(url("class")).catch(() => null) : null,
            on("tgUns") ? load(url("unsure")).catch(() => null) : null]);
        if (id !== want || v !== version)
            return;
        predImg.hidden = !p;
        if (p) {
            predImg.src = p.src;
            predImg.style.opacity = String(+$("predalpha").value / 100);
        }
        unsImg.hidden = !u;
        if (u)
            unsImg.src = u.src;
    }
    async function show(nz) {
        if (!info)
            return;
        z = Math.max(0, Math.min(info.nz - 1, Math.round(nz)));
        $("z").value = String(z);
        $("zlab").textContent = `z ${z}`;
        for (const c of document.querySelectorAll(".chip"))
            c.classList.toggle("on", +c.dataset.z === z);
        // (the previous overlay stays until this slice's prediction has loaded: no dark/light flash)
        plot();
        const id = ++want, q = `run=${encodeURIComponent(info.run)}&z=${z}`;
        const [g, p] = await Promise.all([load(`/api/ent/slice?${q}`), on("tgOv") ? load(`/api/ent/pca?${q}`) : Promise.resolve(null)]);
        if (id !== want)
            return;
        gray = g;
        pca = p;
        sizeLayers(g.naturalWidth, g.naturalHeight);
        drawBase();
        refreshAnnot();
        refreshPred();
        refreshDense();
        refreshNni();
    }
    // ------------------------------------------------------------ annotation tools
    let drawing = false, panning = false, spaceDown = false, lastPt = null;
    let lasso = [], panStart = [0, 0, 0, 0];
    const radius = () => +$("size").value / 2;
    function snapshot() {
        undo.push({ z, data: lab().slice() });
        if (undo.length > 30)
            undo.shift();
    }
    function stamp(x, y, k) {
        const a = lab(), r = radius(), r2 = r * r;
        const x0 = Math.floor(x - r), x1 = Math.ceil(x + r), y0 = Math.floor(y - r), y1 = Math.ceil(y + r);
        for (let yy = Math.max(0, y0); yy < Math.min(H, y1 + 1); yy++)
            for (let xx = Math.max(0, x0); xx < Math.min(W, x1 + 1); xx++) {
                if ((xx - x) ** 2 + (yy - y) ** 2 <= r2)
                    a[yy * W + xx] = k;
            }
        paintBuf(x0, y0, x1 + 1, y1 + 1);
    }
    function strokeTo(p, k) {
        const q = lastPt ?? p, d = Math.hypot(p[0] - q[0], p[1] - q[1]), n = Math.max(1, Math.ceil(d / Math.max(1, radius() / 3)));
        for (let i = 1; i <= n; i++)
            stamp(q[0] + ((p[0] - q[0]) * i) / n, q[1] + ((p[1] - q[1]) * i) / n, k);
    }
    function fillMask(mask, k, subtract) {
        const a = lab();
        for (let i = 0; i < a.length; i++)
            if (mask[i * 4 + 3] > 127)
                a[i] = subtract ? (a[i] === k ? 0 : a[i]) : k;
        paintBuf();
        counts();
        edited();
    }
    function fillLasso(pts, k, subtract) {
        if (pts.length < 3)
            return;
        const off = document.createElement("canvas");
        off.width = W;
        off.height = H;
        const o = off.getContext("2d");
        o.beginPath();
        o.moveTo(pts[0][0], pts[0][1]);
        for (const p of pts.slice(1))
            o.lineTo(p[0], p[1]);
        o.closePath();
        o.fillStyle = "#000";
        o.fill();
        fillMask(o.getImageData(0, 0, W, H).data, k, subtract);
    }
    async function wand(x, y, subtract) {
        if (!info)
            return;
        const thr = +$("thr").value / 100;
        const im = await load(`/api/seg/wand?run=${encodeURIComponent(info.run)}&z=${z}&x=${x}&y=${y}&thr=${thr}`);
        const off = document.createElement("canvas");
        off.width = W;
        off.height = H;
        const o = off.getContext("2d");
        o.imageSmoothingEnabled = true;
        o.drawImage(im, 0, 0, W, H);
        const d = o.getImageData(0, 0, W, H).data;
        for (let i = 0; i < d.length; i += 4)
            d[i + 3] = d[i]; // gray mask -> alpha
        snapshot();
        fillMask(d, cur, subtract);
    }
    function drawCursor(p) {
        cctx.clearRect(0, 0, W, H);
        if (lasso.length > 1) {
            cctx.beginPath();
            cctx.moveTo(lasso[0][0], lasso[0][1]);
            for (const q of lasso)
                cctx.lineTo(q[0], q[1]);
            cctx.strokeStyle = classes[cur - 1]?.color ?? "#fff";
            cctx.lineWidth = 2 / scale;
            cctx.setLineDash([6 / scale, 4 / scale]);
            cctx.stroke();
            cctx.setLineDash([]);
        }
        if (!p || (tool !== "brush" && tool !== "erase"))
            return;
        cctx.beginPath();
        cctx.arc(p[0], p[1], radius(), 0, Math.PI * 2);
        cctx.lineWidth = 1.5 / scale;
        cctx.strokeStyle = tool === "erase" ? "#fff" : classes[cur - 1]?.color ?? "#fff";
        cctx.stroke();
    }
    stage.addEventListener("pointerdown", (ev) => {
        if (ev.target.closest(".tools, .classes, .toast, .gallery, .nnipanel"))
            return;
        stage.setPointerCapture(ev.pointerId);
        const p = toImg(ev);
        if (tool === "pan" || spaceDown || ev.button === 1) {
            panning = true;
            panStart = [ev.clientX, ev.clientY, tx, ty];
            stage.classList.add("panning");
            return;
        }
        if (!classes.length)
            return;
        if (tool === "wand") {
            wand(p[0], p[1], ev.altKey);
            return;
        }
        if (tool === "nni") {
            nniStart = p;
            nniAlt = ev.altKey;
            drawing = true;
            return;
        }
        snapshot();
        drawing = true;
        lastPt = null;
        if (tool === "lasso") {
            lasso = [p];
            return;
        }
        strokeTo(p, tool === "erase" ? 0 : cur);
        lastPt = p;
    });
    stage.addEventListener("pointermove", (ev) => {
        if (inGallery(ev) && !drawing && !panning) {
            drawCursor(null);
            return;
        }
        const p = toImg(ev);
        if (panning) {
            tx = panStart[2] + ev.clientX - panStart[0];
            ty = panStart[3] + ev.clientY - panStart[1];
            applyView();
            return;
        }
        if (drawing && tool === "nni" && nniStart) {
            drawBox(nniStart, p);
            return;
        }
        if (drawing && tool === "lasso")
            lasso.push(p);
        else if (drawing) {
            strokeTo(p, tool === "erase" ? 0 : cur);
            lastPt = p;
        }
        drawCursor(p);
    });
    const endStroke = (ev) => {
        if (panning) {
            panning = false;
            stage.classList.remove("panning");
            return;
        }
        if (!drawing)
            return;
        drawing = false;
        if (tool === "nni" && nniStart) {
            const a = nniStart;
            nniStart = null;
            nniRelease(a, toImg(ev));
            return;
        }
        if (tool === "lasso") {
            fillLasso(lasso, cur, ev.altKey);
            lasso = [];
            drawCursor(toImg(ev));
            return;
        }
        lastPt = null;
        counts();
        edited();
    };
    stage.addEventListener("pointerup", endStroke);
    stage.addEventListener("pointercancel", endStroke);
    stage.addEventListener("pointerleave", () => drawCursor(null));
    function setTool(t) {
        tool = t;
        for (const b of document.querySelectorAll(".tools [data-tool]"))
            b.classList.toggle("on", b.dataset.tool === t);
        stage.classList.toggle("pan", t === "pan");
        document.querySelector(".wandopt").hidden = t !== "wand";
        document.querySelector(".size:not(.wandopt)").hidden = !(t === "brush" || t === "erase");
        $("nnipanel").hidden = t !== "nni";
        if (t === "nni")
            nniEnsure();
        drawCursor(null);
    }
    for (const b of document.querySelectorAll(".tools [data-tool]"))
        b.addEventListener("click", () => setTool(b.dataset.tool));
    $("size").addEventListener("input", () => { $("sizev").textContent = $("size").value; });
    $("thr").addEventListener("input", () => { $("thrv").textContent = (+$("thr").value / 100).toFixed(2); });
    function doUndo() {
        const u = undo.pop();
        if (!u)
            return;
        labels.set(u.z, u.data);
        if (u.z !== z)
            show(u.z);
        else
            refreshAnnot();
        edited(u.z);
    }
    $("undo").addEventListener("click", doUndo);
    // ------------------------------------------------------------ classes
    function renderClasses() {
        const ul = $("clist");
        ul.replaceChildren();
        classes.forEach((c) => {
            const li = document.createElement("li");
            li.classList.toggle("on", c.id === cur);
            const sw = document.createElement("span");
            sw.className = "sw";
            sw.style.background = c.color;
            const nm = document.createElement("span");
            nm.className = "nm";
            nm.textContent = `${c.id}  ${c.name}`;
            const ct = document.createElement("span");
            ct.className = "ct";
            ct.id = `ct${c.id}`;
            const ed = document.createElement("button");
            ed.className = "ren";
            ed.title = "rename";
            ed.textContent = "✎";
            li.append(sw, nm, ct, ed);
            li.addEventListener("click", () => {
                if (cur === c.id)
                    return;
                cur = c.id;
                for (const x of ul.children)
                    x.classList.toggle("on", x === li);
                drawCursor(lastPt);
            });
            ed.addEventListener("click", (e) => {
                e.stopPropagation();
                const inp = document.createElement("input");
                inp.value = c.name;
                inp.setAttribute("aria-label", "class name");
                nm.replaceWith(inp);
                inp.focus();
                inp.select();
                let done = false;
                const finish = (keep) => {
                    if (done)
                        return;
                    done = true;
                    if (keep && inp.value.trim()) {
                        c.name = inp.value.trim();
                        dirty.add(z);
                        save();
                    }
                    renderClasses();
                };
                inp.addEventListener("keydown", (ev) => { if (ev.key === "Enter")
                    finish(true); if (ev.key === "Escape")
                    finish(false); ev.stopPropagation(); });
                inp.addEventListener("blur", () => finish(true));
                inp.addEventListener("click", (ev) => ev.stopPropagation());
            });
            ul.appendChild(li);
        });
        counts();
    }
    function counts() {
        const n = new Array(classes.length + 1).fill(0);
        for (const a of labels.values())
            for (let i = 0; i < a.length; i += 7)
                n[a[i]]++; // strided estimate
        classes.forEach((c) => { const e = document.getElementById(`ct${c.id}`); if (e)
            e.textContent = n[c.id] ? `~${(n[c.id] * 7).toLocaleString()}` : ""; });
    }
    $("addcls").addEventListener("click", () => {
        if (classes.length >= 9) {
            toast("Up to 9 classes");
            return;
        }
        const id = classes.length + 1;
        classes.push({ id, name: `class ${id}`, color: PALETTE[id - 1] });
        cur = id;
        renderClasses();
        dirty.add(z);
        save();
    });
    // ------------------------------------------------------------ persistence
    let saveT = 0;
    const dirty = new Set();
    let saving = Promise.resolve(), saveFailed = false;
    function edited(zz = z) { dirty.add(zz); changedSinceFit = true; updateZ(zz); plot(); explicitNext = true; if (cleared)
        setCleared(false); save(); scheduleLive(); }
    function save() { clearTimeout(saveT); saveT = window.setTimeout(flushSave, 400); }
    // saves are chained so they land in order; a failed save must not block later ones (it is retried)
    function flushSave() {
        clearTimeout(saveT);
        if (!info)
            return saving;
        const run = info.run, zs = [...dirty];
        dirty.clear();
        saving = saving.catch(() => undefined).then(async () => {
            const res = await Promise.allSettled(zs.map((zz) => saveSlice(run, zz)));
            const failed = zs.filter((_, i) => res[i].status === "rejected");
            if (failed.length) {
                if (info?.run === run)
                    for (const zz of failed)
                        dirty.add(zz);
                $("livestat").textContent = "could not save scribbles — retrying";
                saveFailed = true;
                clearTimeout(saveT);
                saveT = window.setTimeout(() => flushSave().catch(() => undefined), 2000);
                throw new Error("save failed");
            }
            if (saveFailed) {
                saveFailed = false;
                scheduleLive();
            } // recovered: fit what was waiting
        });
        return saving;
    }
    async function saveSlice(run, zz) {
        {
            const a = labels.get(zz) ?? new Uint8Array(W * H);
            const off = document.createElement("canvas");
            off.width = W;
            off.height = H;
            const o = off.getContext("2d"), d = o.createImageData(W, H);
            for (let i = 0; i < a.length; i++) {
                d.data[i * 4] = d.data[i * 4 + 1] = d.data[i * 4 + 2] = a[i];
                d.data[i * 4 + 3] = 255;
            }
            o.putImageData(d, 0, 0);
            const r = await fetch("/api/seg/annot", { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ run, z: zz, png: off.toDataURL("image/png"), classes }) });
            if (!r.ok)
                throw new Error(`save ${r.status}`);
        }
    }
    async function loadAnnotations(run, w, h) {
        labels.clear();
        undo.length = 0;
        zClasses.clear();
        const r = await (await fetch(`/api/seg/annotations?run=${encodeURIComponent(run)}`)).json();
        // in a project the classes are shared across tomograms: keep the current ones unless this run has its own
        classes = r.classes ?? (PROJECT && classes.length ? classes :
            [{ id: 1, name: "unassigned", color: UNASSIGNED }, { id: 2, name: PROJECT ? "bacterial cell" : "cell", color: PALETTE[1] }]);
        // class 1 is the "unassigned" correction brush (blue; never coloured in the prediction); its name is editable
        classes[0] = { ...classes[0], color: UNASSIGNED };
        cur = Math.min(cur, classes.length) || 1;
        for (const zz of r.slices) {
            const im = await load(`/api/seg/annot?run=${encodeURIComponent(run)}&z=${zz}&t=${Date.now()}`);
            const off = document.createElement("canvas");
            off.width = w;
            off.height = h;
            const o = off.getContext("2d");
            o.drawImage(im, 0, 0);
            const d = o.getImageData(0, 0, w, h).data, a = new Uint8Array(w * h);
            for (let i = 0; i < a.length; i++)
                a[i] = d[i * 4];
            labels.set(zz, a);
            updateZ(zz);
        }
        renderClasses();
        if (r.slices.length)
            toast(`Loaded scribbles on ${r.slices.length} slice${r.slices.length > 1 ? "s" : ""}: ${r.slices.join(", ")}`);
    }
    // ------------------------------------------------------------ segmentation
    // one fit of the classifier on all saved scribbles; quiet = live mode (status line instead of a toast)
    async function fit_(quiet) {
        if (!info)
            return false;
        if (cleared && quiet)
            return false; // a cleared guess only comes back with an edit
        const run = info.run, t0 = performance.now(), st = $("livestat"), ep = epoch;
        try {
            await flushSave();
        }
        catch {
            return false;
        } // shown in the status line; retried automatically
        st.textContent = "updating";
        st.classList.add("busy");
        try {
            const r = await fetch("/api/seg/train", { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ run, n_classes: classes.length, project: PROJECT, prior: priorAuto ? 0 : prior,
                    explicit: explicitNext || !quiet, head }) }).catch(() => null);
            if (!r) {
                st.textContent = "server not reachable";
                return false;
            }
            const j = await r.json().catch(() => ({ detail: `fit failed (${r.status})` }));
            if (!info || info.run !== run || ep !== epoch)
                return false; // tomogram switched or guess cleared meanwhile
            if (!r.ok && j.detail === "guess cleared") {
                applyCleared();
                return false;
            } // cleared (maybe in another tab)
            if (!r.ok) {
                st.textContent = j.detail ?? "fit failed";
                if (!quiet)
                    toast(j.detail ?? "Segmentation failed");
                return false;
            }
            explicitNext = false;
            changedSinceFit = false;
            const first = !segmented;
            if (priorAuto && j.prior_default) {
                prior = j.prior_default;
                showPrior();
            }
            segmented = true;
            version = j.version;
            unsure = j.unsure;
            for (const id of ["tgPred", "tgUns", "export", "clearg"])
                $(id).disabled = false;
            if (first)
                setOn("tgPred", true);
            uchips(j.slices);
            const ns = j.n_slices ?? j.slices.length;
            st.textContent = `${j.head === "unet" ? "U-Net" : j.head === "unet_pu" ? "Positive only" : "MLP"} ${j.fit_seconds != null ? j.fit_seconds.toFixed(1) + " s" : ""} · ${j.n_pixels.toLocaleString()} px · ${ns} slice${ns > 1 ? "s" : ""}` +
                (j.n_tomograms ? ` · ${j.n_tomograms} tomo${j.n_tomograms > 1 ? "s" : ""}` : "") + ` · ${Math.round(performance.now() - t0)} ms`;
            if (!quiet) {
                const vf = j.volume_fraction.map((f, i) => `${classes[i]?.name ?? i + 1} ${(100 * f).toFixed(0)}%`).join(" · ");
                toast(`Fit on ${j.n_pixels.toLocaleString()} px from ${j.slices.length} slice(s), fit accuracy ${(100 * j.train_acc).toFixed(0)}% · volume: ${vf}`, 7000);
            }
            await refreshPred();
            if (!$("gallery").hidden)
                refreshGallery();
            return true;
        }
        finally {
            st.classList.remove("busy");
        }
    }
    async function clearGuess() {
        if (!info)
            return;
        epoch++;
        setCleared(true);
        clearTimeout(liveT);
        explicitNext = false;
        await fetch("/api/seg/clear", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ run: info.run, n_classes: classes.length, project: PROJECT }) }).catch(() => null);
        applyCleared();
    }
    function applyCleared() {
        epoch++;
        cleared = true;
        clearTimeout(liveT);
        segmented = false;
        version++;
        unsure = [];
        predImg.hidden = true;
        unsImg.hidden = true;
        $("uchips").replaceChildren();
        for (const id of ["tgPred", "tgUns", "export"])
            $(id).disabled = true;
        setOn("tgPred", false);
        setOn("tgUns", false);
        $("livestat").textContent = "cleared — paint to predict again";
        if (!$("gallery").hidden)
            refreshGallery();
    }
    // Clear guess = delete the prediction and every scribble / accepted object it was built from (asks first)
    // (no head switch: the U-Net is the classifier)
    $("clearg").addEventListener("click", async () => {
        if (!info)
            return;
        const scope = PROJECT ? "every tomogram of this project" : "this tomogram";
        if (!confirm(`Clear all: delete the prediction, all scribbles and accepted objects on ${scope}? This cannot be undone. Class names are kept.`))
            return;
        clearTimeout(saveT);
        dirty.clear();
        epoch++;
        const r = await fetch("/api/seg/clear_scribbles", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ run: info.run, n_classes: classes.length, project: PROJECT }) }).then((x) => x.json()).catch(() => null);
        labels.clear();
        undo.length = 0;
        zClasses.clear();
        plot();
        refreshAnnot();
        hasDense = false;
        refreshDense();
        applyCleared();
        setCleared(false); // nothing left to predict from; the next strokes start fresh
        $("clearg").disabled = false;
        $("livestat").textContent = "cleared — start painting";
        toast(r ? `Deleted scribbles on ${r.deleted_slices} slice${r.deleted_slices === 1 ? "" : "s"}` : "Could not reach the server");
    });
    window.addEventListener("storage", (e) => {
        if (e.key === `seg.cleared${PROJECT}` && e.newValue === "1" && segmented)
            applyCleared();
    });
    function scheduleLive() {
        clearTimeout(liveT);
        liveT = window.setTimeout(runLive, 250);
    }
    function nPainted() {
        const seen = new Set();
        for (const a of labels.values())
            for (let i = 0; i < a.length; i += 3)
                if (a[i]) {
                    seen.add(a[i]);
                    if (seen.size > 1)
                        return 2;
                }
        return seen.size;
    }
    async function runLive() {
        // single tomogram: both classes must be painted here; in a project the other tomograms' scribbles count too
        // (the server checks the whole project and answers "paint at least two classes" otherwise)
        if (!PROJECT && nPainted() < 2 && !hasDense) {
            $("livestat").textContent = "paint two classes to start";
            await flushSave().catch(() => undefined);
            return;
        }
        if (liveBusy) {
            livePending = true;
            return;
        }
        liveBusy = true;
        try {
            await fit_(true);
        }
        finally {
            liveBusy = false;
            if (livePending) {
                livePending = false;
                runLive();
            }
        }
    }
    // least-sure slices (inside the kept range, spread out, not already annotated) as "annotate here next" chips
    function uchips(annotated) {
        const box = $("uchips");
        box.replaceChildren();
        if (!info || !unsure.length)
            return;
        const sep = Math.max(10, Math.round(info.nz / 20)), picks = [];
        const order = [...unsure.keys()].filter((zz) => zz >= lo && zz < hi && !annotated.includes(zz)).sort((a, b) => unsure[b] - unsure[a]);
        for (const zz of order) {
            if (picks.every((p) => Math.abs(p - zz) >= sep))
                picks.push(zz);
            if (picks.length === 3)
                break;
        }
        for (const zz of picks) {
            const b = document.createElement("button");
            b.className = "chip q";
            b.dataset.z = String(zz);
            b.title = `model least sure here (mean uncertainty ${unsure[zz].toFixed(2)}): annotate next`;
            b.innerHTML = `<span class="n">?</span>z ${zz}`;
            b.addEventListener("click", () => show(zz));
            box.appendChild(b);
        }
    }
    $("tgPred").addEventListener("click", () => { setOn("tgPred", !on("tgPred")); refreshPred(); });
    $("tgUns").addEventListener("click", () => { setOn("tgUns", !on("tgUns")); refreshPred(); });
    // always live (no toggle): every stroke refits
    $("export").addEventListener("click", async () => {
        if (!info)
            return;
        toast("Exporting full-resolution labels…", 20000);
        const r = await fetch("/api/seg/export", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ run: info.run, n_classes: classes.length, project: PROJECT, prior }) });
        const j = await r.json();
        toast(r.ok ? `Saved ${j.path}` : j.detail, 8000);
    });
    // ------------------------------------------------------------ entropy dock
    function gsmooth(h, s) {
        const R = Math.ceil(3 * s);
        return h.map((_, i) => {
            let a = 0, n = 0;
            for (let k = -R; k <= R; k++) {
                const j = i + k;
                if (j < 0 || j >= h.length)
                    continue;
                const w = Math.exp(-(k * k) / (2 * s * s));
                a += w * h[j];
                n += w;
            }
            return a / n;
        });
    }
    function findValleys(c) {
        const n = c.length, kept = c.slice(lo, hi), amp = Math.max(...kept) - Math.min(...kept) || 1, out = [];
        for (let v = Math.max(lo, 1); v < Math.min(hi, n - 1); v++) {
            if (!(c[v] < c[v - 1] && c[v] <= c[v + 1]))
                continue;
            let l = v, lm = v;
            while (l > 0 && c[l - 1] >= c[v]) {
                l--;
                if (c[l] > c[lm])
                    lm = l;
            }
            let r = v, rm = v;
            while (r < n - 1 && c[r + 1] >= c[v]) {
                r++;
                if (c[r] > c[rm])
                    rm = r;
            }
            if (lm === 0 || rm === n - 1 || lm === v || rm === v)
                continue;
            const depth = Math.min(c[lm], c[rm]) - c[v];
            if (depth / amp >= VMIN)
                out.push({ z: v, depth, frac: depth / amp });
        }
        return out.sort((a, b) => b.depth - a.depth);
    }
    function centreBest() {
        const n = curve.length, c = (n - 1) / 2, sd = 0.2 * n, kept = curve.slice(lo, hi);
        const mx = Math.max(...kept), mn = Math.min(...kept);
        let best = lo, bs = -1;
        for (let zz = lo; zz < hi; zz++) {
            const s = Math.exp(-((zz - c) ** 2) / (2 * sd * sd)) * (mx - curve[zz]) / (mx - mn || 1);
            if (s > bs) {
                bs = s;
                best = zz;
            }
        }
        return best;
    }
    function el(tag, a, text) {
        const e = document.createElementNS(NS, tag);
        for (const k in a)
            e.setAttribute(k, String(a[k]));
        if (text !== undefined)
            e.textContent = text;
        svg.appendChild(e);
        return e;
    }
    // x mapping shared with the slider track (range thumbs inset by ~half the thumb width)
    let sx = (v) => v, inv = (px) => px;
    function plot() {
        if (!info || $("plotbox").hidden)
            return;
        const sl = $("z").getBoundingClientRect(), pb = svg.getBoundingClientRect();
        const Wp = pb.width, Hp = pb.height, inset = 8, x0 = sl.left - pb.left + inset, x1 = sl.right - pb.left - inset, n = info.nz - 1;
        sx = (v) => x0 + (v / n) * (x1 - x0);
        inv = (px) => ((px - x0) / (x1 - x0)) * n;
        svg.setAttribute("viewBox", `0 0 ${Wp} ${Hp}`);
        svg.replaceChildren();
        const mn = Math.min(...curve), mx = Math.max(...curve), pad = (mx - mn) * 0.1 || 0.01;
        const sy = (v) => 8 + (1 - (v - mn + pad) / (mx - mn + 2 * pad)) * (Hp - 16);
        el("rect", { x: sx(0), y: 0, width: sx(lo) - sx(0), height: Hp, class: "cut" });
        el("rect", { x: sx(hi - 1), y: 0, width: sx(n) - sx(hi - 1), height: Hp, class: "cut" });
        const d = curve.map((v, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join("");
        el("path", { d: `${d}L${sx(n)},${Hp}L${sx(0)},${Hp}Z`, class: "area" });
        el("path", { d, class: "curve" });
        vals.forEach((v, i) => {
            el("circle", { cx: sx(v.z), cy: sy(curve[v.z]), r: 5, class: "valley" });
            el("text", { x: sx(v.z), y: sy(curve[v.z]) + 18, "text-anchor": "middle", class: "vlabel" }, String(i + 1));
        });
        // manual annotations: one tick per annotated slice along the bottom, split by the classes painted there
        const tickH = 18;
        for (const [zz, cls] of zClasses) {
            const seg = tickH / cls.length;
            cls.forEach((k, j) => el("rect", { x: sx(zz) - 2.5, y: Hp - tickH + j * seg, width: 5, height: seg,
                fill: classes[k - 1]?.color ?? "#888", class: "annotick" }));
        }
        if (zClasses.size)
            el("text", { x: 4, y: Hp - tickH - 3, class: "annolabel" }, `✎ ${zClasses.size} annotated slice${zClasses.size > 1 ? "s" : ""}`);
        el("line", { x1: sx(z), x2: sx(z), y1: 0, y2: Hp, class: "now" });
    }
    let dragging = false;
    const zAt = (ev) => Math.max(0, Math.min((info?.nz ?? 1) - 1, Math.round(inv(ev.clientX - svg.getBoundingClientRect().left))));
    svg.addEventListener("mousedown", (ev) => {
        dragging = true;
        let zz = zAt(ev);
        const near = [...zClasses.keys()].reduce((b, k) => (Math.abs(k - zz) < Math.abs(b - zz) ? k : b), -1e9);
        if (info && Math.abs(near - zz) <= Math.max(1, Math.round((info.nz - 1) / 300)))
            zz = near;
        show(zz);
    });
    window.addEventListener("mouseup", () => { dragging = false; });
    svg.addEventListener("mousemove", (ev) => {
        if (!info)
            return;
        const zz = zAt(ev), tip = $("tip");
        if (dragging)
            show(zz);
        tip.style.display = "block";
        tip.style.left = `${Math.min(ev.clientX - svg.getBoundingClientRect().left + 12, svg.clientWidth - 160)}px`;
        const near = [...zClasses.keys()].reduce((b, k) => (Math.abs(k - zz) < Math.abs(b - zz) ? k : b), -1e9);
        const hit = Math.abs(near - zz) <= Math.max(1, Math.round((info.nz - 1) / 300));
        tip.textContent = `z ${zz} · entropy ${curve[zz].toFixed(3)}` +
            (hit ? ` · ✎ z ${near}: ${zClasses.get(near).map((k) => classes[k - 1]?.name ?? k).join(" + ")}` : "");
    });
    svg.addEventListener("mouseleave", () => { $("tip").style.display = "none"; });
    function chips() {
        const box = $("chips");
        box.replaceChildren();
        const list = vals.length ? vals.map((v, i) => ({ z: v.z, label: `${i + 1}`, title: `valley depth ${Math.round(100 * v.frac)}% of range` }))
            : [{ z: centreBest(), label: "★", title: "no valley between humps: best centre-weighted slice" }];
        for (const c of list) {
            const b = document.createElement("button");
            b.className = "chip";
            b.dataset.z = String(c.z);
            b.title = c.title;
            b.innerHTML = `<span class="n">${c.label}</span>z ${c.z}`;
            b.addEventListener("click", () => show(c.z));
            box.appendChild(b);
        }
    }
    // ------------------------------------------------------------ nnInteractive tool (service on :8771 via /api/nni)
    let nniReady = false, nniRun = "", nniStart = null, nniAlt = false, nniHas = false, nniBusy = false;
    let hasDense = false;
    const post = (url, body) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) }).then(async (r) => { const j = await r.json().catch(() => ({})); if (!r.ok)
        throw new Error(j.detail ?? r.status); return j; });
    const nstat = (t) => { $("nnistat").textContent = t; };
    async function nniEnsure() {
        if (!info)
            return false;
        if (!nniReady) {
            nstat("starting…");
            for (let i = 0; i < 90 && !nniReady; i++) {
                const r = await post("/api/nni/start", {}).catch(() => ({ running: false }));
                if (r.running)
                    nniReady = true;
                else
                    await new Promise((res) => setTimeout(res, 1000));
            }
            if (!nniReady) {
                nstat("service did not start");
                return false;
            }
        }
        if (nniRun !== info.run) {
            nstat("loading tomogram…");
            await post("/api/nni/open", { run: info.run });
            await post("/api/nni/new", {}); // start from a clean object (no prompts left over from before)
            nniRun = info.run;
            nniHas = false;
            refreshNni();
        }
        nstat("ready");
        const c = await (await fetch(`/api/seg/complete?run=${encodeURIComponent(info.run)}`)).json();
        $("ncomplete").checked = c.complete;
        return true;
    }
    function drawBox(a, b) {
        cctx.clearRect(0, 0, W, H);
        cctx.strokeStyle = classes[cur - 1]?.color ?? "#fff";
        cctx.lineWidth = 2 / scale;
        cctx.setLineDash([6 / scale, 4 / scale]);
        cctx.strokeRect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]));
        cctx.setLineDash([]);
    }
    async function nniRelease(a, b) {
        cctx.clearRect(0, 0, W, H);
        if (nniBusy || !(await nniEnsure()))
            return;
        nniBusy = true;
        nstat("thinking…");
        $("stage").classList.add("busy");
        try {
            const clamp = (v, m) => Math.max(0, Math.min(m - 1, Math.round(v)));
            const isBox = Math.abs(b[0] - a[0]) > 6 && Math.abs(b[1] - a[1]) > 6;
            const r = isBox
                ? await post("/api/nni/box", { z, y0: clamp(a[1], H), y1: clamp(b[1], H), x0: clamp(a[0], W), x1: clamp(b[0], W) })
                : await post("/api/nni/point", { z, y: clamp(b[1], H), x: clamp(b[0], W), positive: !nniAlt });
            nniHas = r.voxels > 0;
            nstat(`${r.seconds.toFixed(1)} s`);
            $("nniinfo").textContent = nniHas
                ? `object: ${r.voxels.toLocaleString()} voxels, slices ${r.z_min}–${r.z_max} · ${r.interactions} prompt${r.interactions > 1 ? "s" : ""}. Scroll z to check it; trim z below if it leaks.`
                : "nothing segmented — try a tighter box or a point inside the object";
            if (nniHas) {
                $("nzfrom").placeholder = String(r.z_min);
                $("nzto").placeholder = String(r.z_max);
            }
            for (const id of ["nniaccept", "nnidiscard"])
                $(id).disabled = !nniHas;
            $("nniaccept").textContent = `Accept as ${classes[cur - 1]?.name ?? "class"}`;
            await refreshNni();
        }
        catch (e) {
            nstat(`error: ${e.message}`.slice(0, 60));
        }
        finally {
            nniBusy = false;
            $("stage").classList.remove("busy");
        }
    }
    async function refreshNni() {
        const c = nniC.getContext("2d");
        c.clearRect(0, 0, W, H);
        if (!nniHas || nniRun !== info?.run)
            return;
        const im = await load(`/api/nni/slice?z=${z}&v=${Date.now()}`).catch(() => null);
        if (!im)
            return;
        colorize(c, im, classes[cur - 1]?.color ?? "#ff7a1a", 150, true);
    }
    function colorize(c, im, color, alpha, outline) {
        const off = document.createElement("canvas");
        off.width = W;
        off.height = H;
        const o = off.getContext("2d");
        o.drawImage(im, 0, 0, W, H);
        const d = o.getImageData(0, 0, W, H), px = d.data, [r, g, b] = hex(color);
        for (let i = 0; i < px.length; i += 4) {
            const on = px[i] > 127;
            px[i] = r;
            px[i + 1] = g;
            px[i + 2] = b;
            px[i + 3] = on ? alpha : 0;
        }
        if (outline)
            for (let y = 1; y < H - 1; y++)
                for (let x = 1; x < W - 1; x++) { // bright edge
                    const i = (y * W + x) * 4;
                    if (px[i + 3] && (!px[i + 3 - 4] || !px[i + 3 + 4] || !px[i + 3 - 4 * W] || !px[i + 3 + 4 * W]))
                        px[i + 3] = 255;
                }
        c.putImageData(d, 0, 0);
    }
    async function refreshDense() {
        const c = denseC.getContext("2d");
        c.clearRect(0, 0, W, H);
        if (!info)
            return;
        const im = await load(`/api/seg/dense?run=${encodeURIComponent(info.run)}&z=${z}&t=${Date.now()}`).catch(() => null);
        if (!im)
            return;
        hasDense = true;
        const off = document.createElement("canvas");
        off.width = W;
        off.height = H;
        const o = off.getContext("2d");
        o.drawImage(im, 0, 0, W, H);
        const d = o.getImageData(0, 0, W, H), px = d.data, cols = classes.map((k) => hex(k.color));
        for (let i = 0; i < px.length; i += 4) {
            const k = px[i];
            if (k && cols[k - 1]) {
                px[i] = cols[k - 1][0];
                px[i + 1] = cols[k - 1][1];
                px[i + 2] = cols[k - 1][2];
                px[i + 3] = 255;
            }
            else
                px[i + 3] = 0;
        }
        c.putImageData(d, 0, 0);
    }
    $("nniaccept").addEventListener("click", async () => {
        if (!info)
            return;
        const zf = $("nzfrom").value, zt = $("nzto").value;
        const r = await post("/api/nni/accept", { class_id: cur, z_from: zf === "" ? -1 : +zf, z_to: zt === "" ? -1 : +zt }).catch((e) => { nstat(String(e)); return null; });
        if (!r)
            return;
        nniHas = false;
        hasDense = true;
        for (const id of ["nniaccept", "nnidiscard"])
            $(id).disabled = true;
        $("nzfrom").value = "";
        $("nzto").value = "";
        $("nniinfo").textContent = `accepted ${r.accepted_voxels.toLocaleString()} voxels as ${classes[cur - 1]?.name}; ${r.labelled_voxels.toLocaleString()} labelled in this tomogram. Next object: drag a box.`;
        refreshNni();
        refreshDense();
        explicitNext = true;
        if (cleared)
            setCleared(false);
        scheduleLive();
    });
    $("nnidiscard").addEventListener("click", async () => {
        await post("/api/nni/new", {}).catch(() => null);
        nniHas = false;
        refreshNni();
        for (const id of ["nniaccept", "nnidiscard"])
            $(id).disabled = true;
        $("nniinfo").textContent = "discarded. Drag a box around an object on this slice.";
    });
    $("ncomplete").addEventListener("change", async () => {
        if (!info)
            return;
        await post("/api/seg/complete", { run: info.run, complete: $("ncomplete").checked });
        explicitNext = true;
        scheduleLive();
    });
    let galT = 0;
    async function refreshGallery() {
        if (!PROJECT)
            return;
        clearTimeout(galT);
        galT = window.setTimeout(async () => {
            const cards = await (await fetch(`/api/project/cards?name=${encodeURIComponent(PROJECT)}`)).json();
            const grid = $("grid"), cols = classes.map((c) => c.color.slice(1)).join(",");
            const nAnn = cards.filter((c) => c.annotated).length;
            $("gsub").textContent = segmented
                ? `one classifier from scribbles on ${nAnn} tomogram${nAnn === 1 ? "" : "s"}, applied to all ${cards.length} · suggested slice of each`
                : `${cards.length} tomograms · paint two classes to see the prediction everywhere`;
            grid.replaceChildren();
            for (const c of cards) {
                const card = document.createElement("div");
                card.className = "card";
                card.classList.toggle("cur", c.run === info?.run);
                const img = document.createElement("img");
                img.loading = "lazy";
                img.alt = c.run;
                img.src = `/api/seg/thumb?run=${encodeURIComponent(c.run)}&z=${c.z}&size=320&v=${version}` +
                    (segmented ? `&project=${encodeURIComponent(PROJECT)}&colors=${cols}&prior=${prior.toFixed(4)}` : "");
                const badge = document.createElement("span");
                badge.className = c.annotated ? "badge" : "badge none";
                badge.textContent = c.annotated ? `✎ ${c.annotated} slice${c.annotated > 1 ? "s" : ""}` : "no scribbles";
                const cap = document.createElement("div");
                cap.className = "cap";
                cap.innerHTML = `<span class="nm">${c.i}. ${c.run}</span><span class="z">z ${c.z}</span>`;
                card.append(img, badge, cap);
                card.addEventListener("click", async () => {
                    $("tomo").value = c.run;
                    toggleGallery(false);
                    await open(c.run);
                    show(c.z);
                });
                grid.appendChild(card);
            }
        }, 150);
    }
    function toggleGallery(v) {
        const g = $("gallery"), on_ = v ?? g.hidden;
        g.hidden = !on_;
        setOn("tgGal", on_);
        for (const sel of [".tools", ".classes"])
            document.querySelector(sel).style.visibility = on_ ? "hidden" : "";
        document.querySelector(".dock").hidden = on_; // no slider / entropy graph under the gallery
        if (!on_)
            requestAnimationFrame(() => { fit(); plot(); });
        if (on_)
            refreshGallery();
    }
    $("tgGal").addEventListener("click", () => toggleGallery());
    let priorT = 0;
    document.getElementById("prior")?.addEventListener("input", () => {
        prior = toPrior(+$("prior").value);
        priorAuto = false;
        showPrior();
        try {
            localStorage.setItem(`seg.prior${PROJECT}`, JSON.stringify({ prior, auto: false }));
        }
        catch { /* storage unavailable */ }
        clearTimeout(priorT);
        priorT = window.setTimeout(() => {
            refreshPred();
            if (!$("gallery").hidden)
                refreshGallery();
            if (PROJECT)
                fetch("/api/project/prior", { method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ project: PROJECT, prior }) });
        }, 150);
    });
    document.getElementById("prior")?.addEventListener("dblclick", () => {
        priorAuto = true;
        try {
            localStorage.removeItem(`seg.prior${PROJECT}`);
        }
        catch { /* storage unavailable */ }
        showPrior();
        scheduleLive();
    });
    showPrior();
    // ------------------------------------------------------------ tomograms
    async function open(run) {
        info = await (await fetch(`/api/ent/info?run=${encodeURIComponent(run)}`)).json();
        const i = info;
        $("z").max = String(i.nz - 1);
        $("nzlab").textContent = `/ ${i.nz - 1}`;
        lo = Math.floor(CUT * i.nz);
        hi = Math.max(lo + 1, Math.ceil((1 - CUT) * i.nz));
        curve = gsmooth(i.H_smooth, SSIG);
        vals = findValleys(curve);
        chips();
        // in a project the classifier is shared by all tomograms: keep it (and the layer toggles) when switching
        const keep = !!PROJECT && segmented;
        unsure = [];
        $("uchips").replaceChildren();
        if (!keep) {
            segmented = false;
            version = 0;
            $("livestat").textContent = "";
            for (const id of ["tgPred", "tgUns", "export"])
                $(id).disabled = true;
            setOn("tgPred", false);
            setOn("tgUns", false);
        }
        if (cleared)
            $("livestat").textContent = "cleared — paint to predict again";
        fetch("/api/seg/warm", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run, n_classes: 2 }) });
        try {
            localStorage.setItem(`seg.run${PROJECT}`, run);
        }
        catch { /* storage unavailable */ }
        await loadAnnotations(run, i.shape[2], i.shape[1]);
        nniHas = false;
        hasDense = false;
        if (tool === "nni")
            nniEnsure();
        W = 0;
        await show(vals.length ? vals[0].z : centreBest());
        if (keep && !changedSinceFit) { // same classifier: only this tomogram's "?" chips need computing
            const r = await post("/api/seg/profile", { run, n_classes: classes.length, project: PROJECT, prior }).catch(() => null);
            if (r && info?.run === run) {
                unsure = r.unsure;
                uchips([...zClasses.keys()]);
            }
        }
        else if (keep) {
            if (!liveBusy) {
                liveBusy = true;
                fit_(true).finally(() => { liveBusy = false; });
            } // refresh this tomogram's "?" chips
        }
        else if (labels.size)
            runLive();
    }
    async function init() {
        const items = await (await fetch(`/api/ent/list?${pq.slice(1)}`)).json(), sel = $("tomo");
        if (PROJECT) {
            $("tgGal").hidden = false;
            const p = await (await fetch(`/api/project?name=${encodeURIComponent(PROJECT)}`)).json();
            document.querySelector(".brand").textContent = p.name ?? PROJECT;
            document.title = `${p.name ?? PROJECT} · cell hunt`;
        }
        items.forEach((it, i) => {
            const o = document.createElement("option");
            o.value = it.run;
            o.textContent = PROJECT ? `${i + 1}. ${it.run}` : `${it.run} · ${it.set}`;
            sel.appendChild(o);
        });
        if (!items.length) {
            toast("No tomograms with an entropy cache — run entropy_cache.py", 10000);
            return;
        }
        let saved = "";
        try {
            saved = localStorage.getItem(`seg.run${PROJECT}`) ?? "";
        }
        catch { /* storage unavailable */ }
        sel.value = items.some((i) => i.run === saved) ? saved : items[0].run;
        sel.addEventListener("change", () => open(sel.value));
        await open(sel.value);
        if (PROJECT && !segmented) { // scribbles elsewhere in the project: show the shared prediction right away
            const cards = await (await fetch(`/api/project/cards?name=${encodeURIComponent(PROJECT)}`)).json();
            if (cards.some((c) => c.annotated) && !liveBusy) {
                liveBusy = true;
                fit_(true).finally(() => { liveBusy = false; });
            }
        }
    }
    // ------------------------------------------------------------ controls
    $("z").addEventListener("input", (e) => show(+e.target.value));
    $("tgPlot").addEventListener("click", () => { const v = !on("tgPlot"); setOn("tgPlot", v); $("plotbox").hidden = !v; plot(); });
    $("tgOv").addEventListener("click", () => { setOn("tgOv", !on("tgOv")); show(z); });
    $("tgSet").addEventListener("click", () => { const p = $("settings"); p.hidden = !p.hidden; $("tgSet").setAttribute("aria-expanded", String(!p.hidden)); });
    document.addEventListener("pointerdown", (ev) => { if (!ev.target.closest(".pop"))
        $("settings").hidden = true; });
    for (const id of ["cmin", "cmax", "ovalpha"])
        $(id).addEventListener("input", drawBase);
    $("annalpha").addEventListener("input", () => { annot.style.opacity = String(+$("annalpha").value / 100); });
    $("predalpha").addEventListener("input", () => { predImg.style.opacity = String(+$("predalpha").value / 100); });
    $("zavg").addEventListener("change", refreshPred);
    window.addEventListener("resize", () => { fit(); plot(); });
    window.addEventListener("keydown", (ev) => {
        const t = ev.target;
        if (t.tagName === "INPUT" && t.type !== "range")
            return;
        const k = ev.key.toLowerCase();
        if (!$("gallery").hidden) { // gallery open: only its own keys
            if (ev.key === "Escape" || ev.key.toLowerCase() === "g") {
                toggleGallery(false);
                ev.preventDefault();
            }
            return;
        }
        if (ev.key === " ") {
            spaceDown = true;
            stage.classList.add("pan");
            ev.preventDefault();
            return;
        }
        if ((ev.ctrlKey || ev.metaKey) && k === "z") {
            doUndo();
            ev.preventDefault();
            return;
        }
        const tools = { b: "brush", e: "erase", l: "lasso", w: "wand", h: "pan" };
        if (tools[k])
            setTool(tools[k]);
        else if (/^[1-9]$/.test(k) && +k <= classes.length && !(head === "unet_pu" && k === "1")) {
            cur = +k;
            renderClasses();
        }
        else if (k === "[" || k === "]") {
            const s = $("size");
            s.value = String(+s.value + (k === "]" ? 4 : -4));
            $("sizev").textContent = s.value;
            drawCursor(lastPt);
        }
        else if (k === "p" && segmented) {
            setOn("tgPred", !on("tgPred"));
            refreshPred();
        }
        else if (k === "u" && segmented) {
            setOn("tgUns", !on("tgUns"));
            refreshPred();
        }
        else if (k === "o") {
            setOn("tgOv", !on("tgOv"));
            show(z);
        }
        else if (k === "f")
            fit();
        else if (k === "g" && PROJECT)
            toggleGallery();
        else if (ev.key === "Escape" && !$("gallery").hidden)
            toggleGallery(false);
        else if (ev.key === "ArrowUp" || ev.key === "ArrowRight")
            show(z + (ev.shiftKey ? 10 : 1));
        else if (ev.key === "ArrowDown" || ev.key === "ArrowLeft")
            show(z - (ev.shiftKey ? 10 : 1));
        else
            return;
        ev.preventDefault();
    });
    window.addEventListener("keyup", (ev) => { if (ev.key === " ") {
        spaceDown = false;
        stage.classList.toggle("pan", tool === "pan");
    } });
    setTool("brush");
    init();
})();
//# sourceMappingURL=entropy.js.map