// Comparison map UI logic (Stage 5.4 — Explorer-style re-layout over 5.3). The
// run/poll/CSV path is unchanged; this file drives two synced Leaflet maps of the
// selected label maps with per-class colour-chip toggles (left-click = solo,
// Ctrl/Cmd+click = add/remove), both products' footprints and their overlap drawn,
// and AOI by draw/upload/type/full-overlap default fed back into the same run flow.
// Results are shown as an ECharts Sankey of the mapping probabilities; the affinity
// heatmap and matching table are download-only. See docs/PIPELINE.md, Stage 5.4.
//
// Stage 6c layers a Harmonize/Review MODE SWITCH on top of this app (bottom of the
// file). Review is the investigate-and-decide inspector: two large synced maps, a
// left rail (class-pair dropdowns + patch thumbnails), a right rail (candidate edges
// + multi-select confirm + provenance), and a basemap/year switcher — consuming only
// the 6b /api/review/* endpoints, sharing this app's live state (AOI, products, the
// on-disk reviewed table). See docs/PIPELINE.md, Stage 6.7 / 6c.

const $ = (id) => document.getElementById(id);

let CALIBRATION = null; // filled from /api/products
let CURRENT_JOB = null;

// --- Split-map state -------------------------------------------------------
const MAPS = { ref: null, cmp: null }; // Leaflet map instances
const LABEL_LAYERS = { ref: null, cmp: null }; // current tile layers
const LEGENDS = { ref: null, cmp: null }; // [{value,name,color}]
const VISIBLE = { ref: null, cmp: null }; // Set of visible class values
const CURRENT_PID = { ref: null, cmp: null }; // product id shown on each side
let OVERLAY_GROUP = null; // {ref,cmp} footprints + overlap outline, mirrored on both maps
let DRAW_LAYER = null; // {ref,cmp} the user-drawn/uploaded AOI rectangle, mirrored on both maps
let SYNCING = false; // guard against view-sync feedback loop

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

// --- AOI cards ---------------------------------------------------------------
// Every AOI of the run is a card in #aoi-list: the primary (the full
// harmonization run) plus the auxiliaries (targeted top-up sampling of the
// classes the primary could not model). Each card owns its bbox inputs, upload,
// overlap check, and Run/Sample button; run PARAMETERS are shared (#param-panel,
// one setting for every AOI). Sampled cards mirror server truth (/api/aoi/list);
// draft cards live only in the page until sampled. Exactly one card is ACTIVE:
// map rectangle drawing and boundary uploads write into it, and its box is the
// dashed rectangle mirrored on both maps.
let AOIS = []; // [{key,isPrimary,name,bbox:[4 strings],sampled,entry,dirty,el,inputs,runBtn,progEl,progMsg,progErr}]
let ACTIVE_AOI = "primary";
let AOI_SEQ = 0;
let SERVER_AOIS = null; // last /api/aoi/list payload (cap, coverage, primary existence)
let RUN_ALL = false; // true while "Run all AOIs" sequences the cards

function primaryCard() {
  return AOIS.find((c) => c.isPrimary);
}

function activeCard() {
  return AOIS.find((c) => c.key === ACTIVE_AOI) || primaryCard();
}

// A card's bbox as numbers: null = all four blank (primary: full overlap);
// throws when partially filled.
function cardBbox(card) {
  const vals = card && card.bbox ? card.bbox.map((v) => String(v).trim()) : ["", "", "", ""];
  if (vals.every((v) => v === "")) return null;
  if (vals.some((v) => v === "")) throw new Error("Enter all four AOI values, or leave all blank.");
  return vals.map(Number);
}

function cardBboxSafe(card) {
  try {
    return cardBbox(card);
  } catch (_) {
    return null;
  }
}

// The RUN AOI (the primary card's box) — what /api/run and the Review workspace
// use. Kept under the old names so the run/review paths read unchanged.
function readAoi() {
  return cardBbox(primaryCard());
}

function readAoiSafe() {
  try {
    return readAoi();
  } catch (_) {
    return null;
  }
}

// The box being edited (the active card's), for the map rectangle + footprints.
function activeBboxSafe() {
  return cardBboxSafe(activeCard());
}

// Write a bbox [minLon,minLat,maxLon,maxLat] into the ACTIVE card (draw/upload).
function setAoiInputs(bbox) {
  const card = activeCard();
  if (card) setCardBbox(card, bbox);
}

function setCardBbox(card, bbox) {
  card.bbox = bbox == null ? ["", "", "", ""] : bbox.map(String);
  card.dirty = true;
  if (card.inputs) card.inputs.forEach((inp, i) => (inp.value = card.bbox[i]));
}

// Per-card progress line. The message is kept on the card object so it survives
// a re-render (refreshAoiList rebuilds the DOM while a job may still be polling).
function setCardProg(card, msg, isError) {
  card.progMsg = msg || "";
  card.progErr = !!isError;
  if (card.progEl) {
    card.progEl.className = "progress" + (isError ? " error" : "");
    card.progEl.textContent = card.progMsg;
  }
}

function selectedPair() {
  return { reference_id: $("reference").value, compare_id: $("compare").value };
}

// --- Init: populate product dropdowns and default run parameters. ----------

async function init() {
  const data = await getJSON("/api/products");
  CALIBRATION = data.calibration;

  // Every label product is offered on BOTH sides — role is advisory metadata,
  // not an enforced pairing constraint (any product may be reference or
  // compare), so the two dropdowns must list the same set. Each option is
  // tagged with its source (local raster vs GEE) so the user can tell them
  // apart before picking.
  const labels = data.products.filter((p) => p.kind === "label");
  fillSelect($("reference"), labels);
  fillSelect($("compare"), labels);

  $("year-note").textContent = `working year ${data.working_year}`;
  $("sample_scale_m").value = data.defaults.sample_scale_m;
  $("n_components").value = data.defaults.n_components;
  $("points_floor").value = data.defaults.points_floor;
  $("points_target").value = data.defaults.points_target;

  if (data.review) {
    $("rev-n").value = data.review.patches_per_pair;
    $("rev-oversample").value = data.review.live_oversample;
  }

  const c = data.calibration;
  $("calibration-note").textContent =
    `Calibration (live from config): temperature T=${c.softmax_temperature}, ` +
    `affinity floor=${c.absolute_affinity_floor ?? "None (uncalibrated)"}, ` +
    `margin threshold=${c.margin_threshold}.`;

  // Stage 8d: anchor the alpha slider on the calibrated config value. ALPHA
  // stays null so a session that never touches the slider follows config.
  ALPHA_DEFAULT = c.semantic_prior_alpha ?? 1;
  wireAlpha();

  // Stand up the split map, then load the two selected label maps onto it.
  setupMaps();
  await refreshMapSide("ref");
  await refreshMapSide("cmp");
  await refreshFootprints();

  // A dataset dropped into data/ registers itself on server startup; watch it
  // through to selectable without the user reloading the page (DESIGN.md 4.1).
  pollDatasetsUntilReady();

  // ...and say so on load when a dataset is waiting for something from the user.
  // A greyed row in a dropdown is not a prompt: the folder is there, the app has
  // noticed it, and it is blocked on one file the user has to supply -- that has
  // to be stated, not left to be discovered by opening the picker.
  await announceBlockedDatasets();
}

// Tell the user, unprompted, about datasets that cannot proceed without them.
// Only ever states a fact and what to do about it; it never modifies data/.
async function announceBlockedDatasets() {
  let data;
  try {
    data = await getJSON("/api/datasets");
  } catch (_) {
    return;
  }
  const blocked = (data.datasets || []).filter(
    (d) => d.state === "needs-legend" || d.state === "error"
  );
  if (!blocked.length) return;

  const status = $("map-status");
  status.className = "note warn";
  const names = blocked.map((d) => `“${d.folder}”`).join(", ");
  status.textContent =
    blocked.length === 1
      ? `${names} needs a legend before it can be used — see the panel below.`
      : `${blocked.length} datasets need attention: ${names} — see the panel below.`;

  // The detail (exact path, required columns, the layout rule) goes in the
  // info panel, which has room for it.
  const needLegend = blocked.filter((d) => d.state === "needs-legend");
  if (needLegend.length) {
    await showDropInRules(needLegend);
  } else {
    const info = $("overlap-info");
    info.className = "info error";
    info.style.whiteSpace = "pre-wrap";
    info.textContent = blocked
      .map((d) => `${d.folder}: ${d.detail}`)
      .join("\n\n");
  }
}

// "Refresh datasets": rescan data/ and start registering anything new. The
// counterpart of dropping a folder in while the server is already running.
async function refreshDatasets() {
  const status = $("map-status");
  status.className = "note";
  status.textContent = "Rescanning data/ …";
  try {
    const r = await getJSON("/api/datasets/refresh", { method: "POST" });
    const started = r.started || [];
    const pending = (r.datasets || []).filter((d) => d.state === "needs-legend");

    // A dataset folder that has been deleted leaves its derived files behind
    // (a converted COG tree is often several GB). Offer to remove them, but
    // only ever on an explicit confirmation — a rescan silently deleting
    // gigabytes is exactly the surprise this flow exists to avoid.
    const missing = (r.datasets || []).filter((d) => d.state === "missing");
    if (missing.length) {
      await offerCleanup(missing);
      return;
    }

    if (started.length) {
      status.textContent = `Registering ${started.length} dataset(s): ${started
        .map((j) => j.folder)
        .join(", ")}`;
      pollDatasetsUntilReady();
    } else if (pending.length) {
      status.className = "note warn";
      status.textContent =
        `Nothing to register. ${pending.length} dataset(s) still need a legend CSV: ` +
        pending.map((d) => d.folder).join(", ");
      // Teach the convention at the point of failure — the rules come from the
      // server so this text cannot drift from the code that enforces them.
      showDropInRules(pending);
    } else {
      status.textContent = "All datasets are up to date.";
    }
  } catch (e) {
    status.className = "note error";
    status.textContent = "Could not refresh datasets: " + e.message;
  }
}

// Ready states a local product can be in (DESIGN.md 4.2). Only "ready" is
// selectable; the rest are shown, disabled, with the reason — a product that is
// still indexing must not silently vanish from the picker, or the user cannot
// tell "not supported" from "not finished yet".
const STATE_LABEL = {
  ready: "",
  indexing: "indexing…",
  converting: "converting…",
  "needs-legend": "needs legend",
  "needs-conversion": "needs conversion",
  error: "error",
  missing: "data folder deleted",
};

// Grouped picker: local datasets first (this is a local-first app), then GEE.
// Everything shown comes from /api/products — no product knowledge is hardcoded
// here, per PIPELINE.md 2.5.
function fillSelect(sel, products) {
  const previous = sel.value;
  sel.innerHTML = "";

  const groups = [
    ["Local datasets", products.filter((p) => p.source === "local_raster")],
    ["GEE datasets", products.filter((p) => p.source !== "local_raster")],
  ];

  for (const [label, items] of groups) {
    if (!items.length) continue;
    const group = document.createElement("optgroup");
    group.label = label;
    for (const p of items) {
      const opt = document.createElement("option");
      opt.value = p.id;

      // Facts that help tell two similar maps apart, from the registry. Kept
      // SHORT: this text also renders in the closed selector, which is a narrow
      // floating control over the map. Dynamic World declares ten available
      // years — spelling them all out pushed its row past 100 characters and
      // truncated every name with an ellipsis.
      const bits = [];
      const years = p.years || [];
      if (years.length === 1) bits.push(String(years[0]));
      else if (years.length > 1) bits.push(`${years[0]}–${years[years.length - 1]}`);
      if (p.resolution_m) bits.push(`${Math.round(p.resolution_m)} m`);
      const meta = bits.length ? ` · ${bits.join(", ")}` : "";

      const state = p.state || "ready";
      const badge = STATE_LABEL[state] ? `  [${STATE_LABEL[state]}]` : "";
      opt.textContent = `${p.name}${meta}${badge}`;
      // The full detail lives in the tooltip, where length costs nothing.
      opt.title = years.length
        ? `${p.name} — years: ${years.join(", ")}`
        : p.name;

      if (state !== "ready") {
        opt.disabled = true;
        opt.title = p.state_detail || STATE_LABEL[state] || state;
      }
      group.appendChild(opt);
    }
    sel.appendChild(group);
  }

  // Keep the current selection across refreshes, but never land on a product
  // that is no longer selectable.
  const stillValid = [...sel.options].some(
    (o) => o.value === previous && !o.disabled
  );
  if (stillValid) sel.value = previous;
  else {
    const first = [...sel.options].find((o) => !o.disabled);
    if (first) sel.value = first.value;
  }
}

// Poll /api/products while any local dataset is still registering, so a product
// that starts as "indexing…" becomes selectable on its own. Stops as soon as
// nothing is in flight — this is a startup-time activity, not a heartbeat.
let DATASET_POLL = null;

async function pollDatasetsUntilReady() {
  if (DATASET_POLL) return;
  const tick = async () => {
    try {
      const { datasets, active } = await getJSON("/api/datasets");
      const busy = datasets.filter(
        (d) => d.state === "indexing" || d.state === "converting"
      );
      const status = $("map-status");
      if (busy.length) {
        const one = busy[0];
        status.className = "note";
        status.textContent =
          `Preparing ${busy.length} dataset${busy.length > 1 ? "s" : ""}: ` +
          `${one.folder} ${STATE_LABEL[one.state]} ` +
          `${Math.round((one.progress || 0) * 100)}%`;
      }
      // Re-fill the dropdowns so newly-ready products become selectable.
      const data = await getJSON("/api/products");
      const labels = data.products.filter((p) => p.kind === "label");
      fillSelect($("reference"), labels);
      fillSelect($("compare"), labels);

      if (!active && !busy.length) {
        clearInterval(DATASET_POLL);
        DATASET_POLL = null;
        if (status.textContent.startsWith("Preparing ")) status.textContent = "";
      }
    } catch (_) {
      clearInterval(DATASET_POLL);
      DATASET_POLL = null;
    }
  };
  DATASET_POLL = setInterval(tick, 3000);
  tick();
}

// --- Split map: setup, tiles, toggles, footprints, AOI draw ----------------

function setupMaps() {
  const world = [[-85, -180], [85, 180]];
  MAPS.ref = L.map("map-ref", { worldCopyJump: true }).setView([16, 32], 3);
  MAPS.cmp = L.map("map-cmp", { worldCopyJump: true }).setView([16, 32], 3);
  const baseUrl = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
  const attrib = "© OpenStreetMap contributors";
  L.tileLayer(baseUrl, { attribution: attrib, maxZoom: 18 }).addTo(MAPS.ref);
  L.tileLayer(baseUrl, { attribution: attrib, maxZoom: 18 }).addTo(MAPS.cmp);

  // Overlay group (footprints + overlap) is mirrored onto both maps.
  OVERLAY_GROUP = { ref: L.layerGroup().addTo(MAPS.ref), cmp: L.layerGroup().addTo(MAPS.cmp) };

  // Keep the two maps in lock-step so a class in one lines up with the other.
  syncPair(MAPS.ref, MAPS.cmp);
  syncPair(MAPS.cmp, MAPS.ref);

  // Draw control (rectangle only) on BOTH maps sets the AOI.
  addDrawControl(MAPS.ref);
  addDrawControl(MAPS.cmp);
}

// Add a rectangle-only AOI draw control to a map; a drawn rectangle sets the AOI.
function addDrawControl(map) {
  const drawn = new L.FeatureGroup().addTo(map);
  const drawControl = new L.Control.Draw({
    draw: {
      rectangle: { shapeOptions: { color: "#c53030", weight: 2 } },
      polygon: false, polyline: false, circle: false,
      marker: false, circlemarker: false,
    },
    edit: { featureGroup: drawn, edit: false, remove: false },
  });
  map.addControl(drawControl);
  map.on(L.Draw.Event.CREATED, (e) => {
    drawn.clearLayers();
    setDrawnAoiFromLayer(e.layer);
  });
}

function syncPair(src, dst) {
  src.on("move", () => {
    if (SYNCING) return;
    SYNCING = true;
    dst.setView(src.getCenter(), src.getZoom(), { animate: false });
    SYNCING = false;
  });
}

// Reflect a drawn/uploaded AOI rectangle into inputs + the map overlay.
function setDrawnAoiFromLayer(layer) {
  const b = layer.getBounds();
  const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map(
    (x) => Number(x.toFixed(5))
  );
  setAoiInputs(bbox);
  drawAoiRect(bbox);
  refreshFootprints();
}

function drawAoiRect(bbox) {
  if (DRAW_LAYER) {
    MAPS.ref.removeLayer(DRAW_LAYER.ref);
    MAPS.cmp.removeLayer(DRAW_LAYER.cmp);
    DRAW_LAYER = null;
  }
  if (!bbox) return;
  const [w, s, e, n] = bbox;
  const style = { color: "#c53030", weight: 2, fill: false, dashArray: "4 3" };
  DRAW_LAYER = {
    ref: L.rectangle([[s, w], [n, e]], style).addTo(MAPS.ref),
    cmp: L.rectangle([[s, w], [n, e]], style).addTo(MAPS.cmp),
  };
}

// Load (or reload) one side's label tiles + legend for its selected product.
async function refreshMapSide(side) {
  const pid = side === "ref" ? $("reference").value : $("compare").value;
  const legendEl = $(side === "ref" ? "legend-ref" : "legend-cmp");
  CURRENT_PID[side] = pid;

  // Legend first: it drives the toggles and the initial "all visible" set. The
  // legend container keeps its height even when empty so both maps stay aligned.
  let legend;
  try {
    legend = (await getJSON(`/api/legend/${pid}`)).classes;
  } catch (e) {
    LEGENDS[side] = null;
    legendEl.innerHTML = `<span class="empty">No legend for this product.</span>`;
    return;
  }
  LEGENDS[side] = legend;
  // Classes the data does not contain are never "visible": they have no pixels
  // to paint, and their chips are non-toggleable, so including them would leave
  // a class in the visible set with no way to remove it.
  VISIBLE[side] = new Set(
    legend.filter((c) => c.observed !== false).map((c) => c.value)
  );
  renderLegend(side);
  await loadTiles(side);
}

function productName(pid) {
  for (const s of [$("reference"), $("compare")]) {
    for (const o of s.options) if (o.value === pid) return o.textContent;
  }
  return pid;
}

// Colour-chip legend: left-click shows only that class; Ctrl/Cmd+click adds or
// removes the class from the visible set (faded chip = hidden).
function renderLegend(side) {
  const legendEl = $(side === "ref" ? "legend-ref" : "legend-cmp");
  legendEl.innerHTML = "";
  for (const c of LEGENDS[side]) {
    // `observed === false` means the indexer scanned this dataset's pixels and
    // did not find the class (DESIGN.md 4.3). It stays in the legend — a
    // regional subset legitimately lacks classes of a global legend — but it is
    // greyed and non-toggleable, because there is nothing to show or hide.
    // `undefined`/`null` means "not determined" (GEE products, hand-written
    // entries), which must behave exactly as before.
    const absent = c.observed === false;
    const on = VISIBLE[side].has(c.value);
    const chip = document.createElement("div");
    chip.className =
      "legend-chip" + (absent ? " absent" : on ? "" : " off");
    chip.title = absent
      ? `Not present in this dataset — declared by the legend, but no pixels of this class were found when it was indexed.${
          c.description ? "\n\n" + c.description : ""
        }`
      : c.description
      ? c.description
      : "Click: show only this class · Ctrl/Cmd+click: add/remove";
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = "#" + c.color;
    const txt = document.createElement("span");
    txt.textContent = `${c.name} (${c.value})`;
    chip.appendChild(sw);
    chip.appendChild(txt);
    if (!absent) {
      chip.addEventListener("click", (e) =>
        selectClass(side, c.value, e.ctrlKey || e.metaKey)
      );
    }
    legendEl.appendChild(chip);
  }
}

// additive=false → solo this class; additive=true → toggle it in/out of the set.
function selectClass(side, value, additive) {
  const set = VISIBLE[side];
  if (additive) {
    if (set.has(value)) set.delete(value);
    else set.add(value);
  } else {
    VISIBLE[side] = new Set([value]);
  }
  renderLegend(side); // re-tint the chips
  // Class-code layers recolour the tiles they already hold — no network at all.
  // Only the server-coloured fallback has to re-fetch.
  const layer = LABEL_LAYERS[side];
  if (layer && layer.recolor) layer.recolor(VISIBLE[side]);
  else loadTiles(side);
}

// --- Class-code tile layer ---------------------------------------------------
// The local-raster tile endpoint serves ONE tile per position encoding each
// pixel's raw class code (greyscale) plus alpha for nodata — no palette, no
// class subset (DESIGN.md 2.3). This layer decodes that tile into a canvas and
// applies the legend palette itself, which is what makes toggling classes free:
// `recolor()` re-runs a per-tile canvas pass over tiles already in memory, with
// no request, no re-render, and no cache churn. One fetched tile therefore
// serves every toggle state.
//
// Colours come from the legend the server already sent, so the palette is still
// registry-driven (PIPELINE.md 2.5) — nothing about the classes is hardcoded here.
const ClassCodeLayer = L.GridLayer.extend({
  // opts: { template, palette: Map(code -> [r,g,b]), visible: Set(code) }
  initialize: function (opts) {
    L.GridLayer.prototype.initialize.call(this, opts);
    this._palette = opts.palette;
    this._visible = opts.visible;
    // Decoded codes/alpha per tile key, kept so recolouring needs no re-decode
    // and no re-fetch. Dropped in `tileunload` with Leaflet's own tile eviction,
    // so this cannot grow without bound as the user pans.
    this._decoded = new Map();
    this.on("tileunload", (e) => {
      this._decoded.delete(this._tileKey(e.coords));
    });
  },

  _tileKey: function (c) {
    return `${c.z}/${c.x}/${c.y}`;
  },

  createTile: function (coords, done) {
    const size = this.getTileSize();
    const canvas = L.DomUtil.create("canvas", "leaflet-tile");
    canvas.width = size.x;
    canvas.height = size.y;
    const key = this._tileKey(coords);

    const url = L.Util.template(this.options.template, {
      z: coords.z,
      x: coords.x,
      y: coords.y,
    });

    const img = new Image();
    // Same-origin, but the canvas must stay untainted for getImageData().
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        // Read the codes back out of the PNG. The tile is lossless 8-bit
        // greyscale+alpha, so the values here are the class codes verbatim —
        // this is why the encoding must never scale or offset them.
        const off = document.createElement("canvas");
        off.width = size.x;
        off.height = size.y;
        const octx = off.getContext("2d", { willReadFrequently: true });
        octx.drawImage(img, 0, 0);
        const raw = octx.getImageData(0, 0, size.x, size.y).data;
        // Greyscale unpacks to r=g=b=code; alpha rides in the 4th byte.
        const n = size.x * size.y;
        const codes = new Uint8Array(n);
        const alpha = new Uint8Array(n);
        for (let i = 0; i < n; i++) {
          codes[i] = raw[i * 4];
          alpha[i] = raw[i * 4 + 3];
        }
        this._decoded.set(key, { codes, alpha });
        this._paint(canvas, codes, alpha);
        done(null, canvas);
      } catch (err) {
        done(err, canvas);
      }
    };
    img.onerror = () => {
      // A tile outside the product's footprint 404s. That is ordinary — leave
      // the canvas blank rather than reporting a layer error.
      this._decoded.delete(key);
      done(null, canvas);
    };
    img.src = url;
    return canvas;
  },

  // Paint decoded codes through the palette. Hidden classes, unknown codes and
  // nodata all become fully transparent so the basemap shows through.
  _paint: function (canvas, codes, alpha) {
    const ctx = canvas.getContext("2d");
    const out = ctx.createImageData(canvas.width, canvas.height);
    const d = out.data;
    // Flat lookup tables beat a Map hit per pixel: 65k pixels a tile, and this
    // runs for every visible tile on every toggle.
    const lut = this._lut();
    for (let i = 0; i < codes.length; i++) {
      const c = codes[i] * 4;
      const visible = alpha[i] !== 0 && lut[c + 3] !== 0;
      const j = i * 4;
      if (visible) {
        d[j] = lut[c];
        d[j + 1] = lut[c + 1];
        d[j + 2] = lut[c + 2];
        d[j + 3] = 255;
      } else {
        d[j + 3] = 0;
      }
    }
    ctx.putImageData(out, 0, 0);
  },

  // 256-entry RGBA lookup, rebuilt only when the visible set changes.
  _lut: function () {
    if (this._lutCache) return this._lutCache;
    const lut = new Uint8ClampedArray(256 * 4);
    for (const [code, rgb] of this._palette) {
      if (code < 0 || code > 255) continue; // not encodable in an 8-bit tile
      if (!this._visible.has(code)) continue;
      const o = code * 4;
      lut[o] = rgb[0];
      lut[o + 1] = rgb[1];
      lut[o + 2] = rgb[2];
      lut[o + 3] = 255; // marks "this code is painted"
    }
    this._lutCache = lut;
    return lut;
  },

  // Re-colour every tile in place for a new visible set. No network.
  recolor: function (visible) {
    this._visible = visible;
    this._lutCache = null;
    for (const key of Object.keys(this._tiles)) {
      const tile = this._tiles[key];
      const decoded = this._decoded.get(this._tileKey(tile.coords));
      if (decoded) this._paint(tile.el, decoded.codes, decoded.alpha);
    }
  },
});

// "RRGGBB" → [r,g,b]
function hexToRgb(hex) {
  const h = String(hex).replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

// Fetch the XYZ tile template for the visible subset and swap the layer in.
async function loadTiles(side) {
  const pid = CURRENT_PID[side];
  const map = MAPS[side];
  const status = $("map-status");
  const values = [...VISIBLE[side]];
  const qs = values.length ? `?classes=${values.join(",")}` : "?classes=";
  status.textContent = `Loading ${productName(pid)} tiles…`;
  try {
    const {
      template,
      max_native_zoom: maxNativeZoom,
      bounds,
      encoding,
    } = await getJSON(`/api/tiles/${pid}${qs}`);

    // Add the new layer *before* removing the old one and only drop the old one
    // once the new tiles have actually arrived. Removing first left the map
    // blank for the whole fetch, which is what made switching maps feel slow
    // even when the tiles themselves were quick.
    const previous = LABEL_LAYERS[side];
    const common = {
      opacity: 0.85,
      // Keep a ring of off-screen tiles so a small pan re-uses them instead of
      // re-requesting, and hold already-drawn tiles while new ones load. Kept
      // deliberately small: eager prefetching pays off only against a fast
      // backend, and while a tile can cost seconds it is actively harmful — a
      // drag queues dozens of requests for viewports the user has already left,
      // and the tiles they *are* looking at wait behind them. Worth revisiting
      // upwards once every product is COG-converted.
      keepBuffer: 2,
      updateWhenIdle: true,
      updateWhenZooming: false,
      // Past the data's real resolution the browser upscales tiles it already
      // has rather than requesting finer ones the raster cannot fill. The cap
      // comes from the product's own metadata, not a fixed number, because
      // these products differ in ground resolution.
      ...(maxNativeZoom != null ? { maxNativeZoom, maxZoom: 22 } : {}),
      // Confine requests to the product's own footprint. A regional map cannot
      // serve a tile outside its extent, and Leaflet would otherwise request
      // the whole viewport and collect a 404 for each one — correct behaviour
      // on the server, but hundreds of console errors that hide real problems.
      // [minLon, minLat, maxLon, maxLat] → Leaflet's [[south, west], [north, east]].
      ...(bounds
        ? { bounds: L.latLngBounds([bounds[1], bounds[0]], [bounds[3], bounds[2]]) }
        : {}),
      className: "label-layer",
    };

    // Local-raster products serve class codes and are coloured here, so a class
    // toggle never reaches the network (see ClassCodeLayer). GEE products stream
    // ready-coloured tiles straight from Earth Engine, where the subset is baked
    // in server-side and a toggle still means a new template.
    let layer;
    if (encoding === "class_code") {
      const palette = new Map(
        (LEGENDS[side] || []).map((c) => [c.value, hexToRgb(c.color)])
      );
      layer = new ClassCodeLayer({
        ...common,
        template,
        palette,
        visible: VISIBLE[side],
      });
    } else {
      layer = L.tileLayer(template, common);
    }

    let swapped = false;
    const dropPrevious = () => {
      if (swapped) return;
      swapped = true;
      if (previous && map.hasLayer(previous)) map.removeLayer(previous);
      status.textContent = "";
    };
    // Swap on 'load' (every visible tile has arrived). 'load' does not fire when
    // Leaflet serves the whole viewport from its own cache, so 'tileload' is a
    // second trigger: once any tile of the new layer has actually painted, the
    // old one is safe to remove.
    //
    // Deliberately NOT a bare timeout. Dropping the previous layer after a fixed
    // delay regardless of whether the new tiles had arrived is what made the map
    // "sometimes show, sometimes not": on a slow first render the old layer went
    // away and nothing had replaced it yet.
    layer.on("load", dropPrevious);
    layer.once("tileload", dropPrevious);
    // Panned entirely outside this product's extent: every tile errors, so
    // neither event above fires. Still clear the status, or it reads "Loading…"
    // forever over a view the product simply does not cover.
    layer.on("tileerror", () => {
      status.textContent = "";
    });

    LABEL_LAYERS[side] = layer.addTo(map);

    // The current view may lie entirely OUTSIDE the new product's footprint --
    // switching from an Africa map to a Southeast Asia one, say. Leaflet's
    // `bounds` option then creates **no tiles at all**, so neither 'load' nor
    // 'tileload' nor 'tileerror' ever fires and the swap above would never run:
    // the previous product's tiles stayed on screen until the user zoomed out
    // far enough to intersect the new footprint, which looked like "the map does
    // not update when I change product".
    //
    // Retiring the old layer is not enough on its own -- that just leaves an
    // empty pane. `refreshFootprints()` re-fits BOTH maps to both products'
    // footprints after every product change, so the new data comes into view by
    // itself; this only has to make sure the stale tiles are gone by then.
    const settleIfEmpty = () => {
      if (swapped) return;
      if (Object.keys(layer._tiles || {}).length === 0) dropPrevious();
    };
    // After the current frame, so Leaflet has run its first _update() pass.
    // One check is enough: this only decides whether to retire the PREVIOUS
    // layer, and by this point Leaflet has already created whatever tiles the
    // current view calls for. Later pans are the new layer's own business.
    setTimeout(settleIfEmpty, 0);
  } catch (e) {
    status.className = "note error";
    status.textContent = `Could not load ${productName(pid)} tiles: ${e.message}`;
  }
}

// Draw both products' footprints and their overlap outline (from the endpoint).
async function refreshFootprints() {
  if (!OVERLAY_GROUP) return;
  const aoi = activeBboxSafe();
  const p = new URLSearchParams(selectedPair());
  if (aoi) {
    p.set("min_lon", aoi[0]); p.set("min_lat", aoi[1]);
    p.set("max_lon", aoi[2]); p.set("max_lat", aoi[3]);
  }
  let data;
  try {
    data = await getJSON(`/api/footprints?${p.toString()}`);
  } catch (e) {
    // Returning silently here used to hide a real failure: for a pair with no
    // common ground (an Africa map beside a Southeast Asia one) the endpoint
    // answered 400, so no footprints were drawn AND the view was never re-fitted
    // — leaving whatever the previous product had painted on screen. The
    // endpoint now reports an empty overlap as data, so reaching this branch
    // means something actually went wrong and should be said out loud.
    const status = $("map-status");
    status.className = "note error";
    status.textContent = `Could not load footprints: ${e.message}`;
    return;
  }
  OVERLAY_GROUP.ref.clearLayers();
  OVERLAY_GROUP.cmp.clearLayers();
  const colors = ["#2b6cb0", "#805ad5"];
  const bounds = [];
  // A layer can only live on one map, so add an independent copy to each side.
  const addBoth = (geometry, style, tooltip) => {
    for (const side of ["ref", "cmp"]) {
      const gj = L.geoJSON(geometry, { style });
      gj.bindTooltip(tooltip);
      gj.addTo(OVERLAY_GROUP[side]);
      if (side === "ref") bounds.push(gj.getBounds());
    }
  };
  data.products.forEach((prod, i) => {
    if (!prod.geometry) return; // global product: no box to draw
    addBoth(prod.geometry, { color: colors[i], weight: 2, fill: false, dashArray: "6 4" },
      `${prod.name} footprint`);
  });
  if (data.overlap && data.overlap.geometry && !data.overlap.is_global) {
    addBoth(data.overlap.geometry,
      { color: "#2f855a", weight: 3, fillColor: "#2f855a", fillOpacity: 0.1 }, "overlap");
  }

  // Fit both maps to everything we drew (both footprints / overlap / AOI). This
  // is what makes a product change *visible*: the two maps may be on opposite
  // sides of the world (Africa 9–45°E vs Southeast Asia 88–144°E), and without a
  // re-fit the pane you just changed shows empty space — or, worse, still shows
  // the previous product's tiles. Zooming out to contain both footprints puts
  // the new data on screen immediately.
  let fit = null;
  for (const b of bounds) fit = fit ? fit.extend(b) : L.latLngBounds(b.getSouthWest(), b.getNorthEast());
  if (DRAW_LAYER) fit = (fit || DRAW_LAYER.ref.getBounds()).extend(DRAW_LAYER.ref.getBounds());
  if (fit && fit.isValid()) {
    SYNCING = true;
    MAPS.ref.fitBounds(fit.pad(0.15));
    MAPS.cmp.fitBounds(fit.pad(0.15));
    SYNCING = false;
  }

  // Say so when the pair has no common ground: the two boxes are now both
  // visible, but there is nothing to run, and the reason should not have to be
  // inferred from the picture.
  if (data.overlap_error) {
    const status = $("map-status");
    status.className = "note warn";
    status.textContent =
      `These two maps do not overlap — ${data.overlap_error}. ` +
      `Both footprints are outlined; pick a pair that shares ground to run.`;
  }
}

// AOI boundary upload (GeoJSON): take its bounding box as the card's rectangle.
async function uploadAoi(file, card) {
  const status = $("map-status");
  try {
    const gj = JSON.parse(await file.text());
    const layer = L.geoJSON(gj);
    const b = layer.getBounds();
    if (!b.isValid()) throw new Error("empty or invalid geometry");
    const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map(
      (x) => Number(x.toFixed(5))
    );
    setCardBbox(card, bbox);
    setActiveCard(card.key);
    drawAoiRect(bbox);
    await refreshFootprints();
    status.className = "note";
    status.textContent = `AOI “${cardLabel(card)}” set from uploaded boundary [${bbox.join(", ")}].`;
  } catch (e) {
    status.className = "note error";
    status.textContent = "Could not read boundary file: " + e.message;
  }
}

function cardLabel(card) {
  return card.isPrimary ? "primary" : card.name && card.name.trim() ? card.name.trim() : "unnamed auxiliary";
}

// --- Overlap check (per card) ----------------------------------------------

// Compute the two products' overlap and ADOPT it as this card's AOI.
//
// The button used to only print the bbox, which read as "nothing happened": the
// card's four inputs stayed blank and the map did not move. The overlap is only
// useful as an AOI, so it is written into the card (setCardBbox), drawn as the
// dashed rectangle, and the maps are fitted to it — the same treatment an
// uploaded boundary gets in uploadAoi().
//
// The AOI is a BOUNDING BOX, while the true overlap of two maps is whatever
// shape their data happens to be (coastlines, missing tiles, no-data corners).
// That is by design, not an approximation being papered over: sampling walks the
// box cell by cell and a cell with no data in a product simply yields no
// candidates there, so the points drawn are always inside real data. The only
// cost of the box being larger than the true overlap is scan time spent on empty
// cells. See DESIGN.md 3.1.
//
// A GLOBAL overlap is the one case that is NOT adopted: it is the whole planet,
// so filling in [-180,-90,180,90] would replace "blank = full overlap" with a
// box that means the same thing while looking like a deliberate choice, and on
// the GEE path it is a hard blocker anyway. Same for an error response.
async function checkOverlap(card) {
  const info = $("overlap-info");
  info.className = "info";
  info.textContent = `Checking overlap for “${cardLabel(card)}”…`;
  try {
    const body = {
      ...selectedPair(),
      // Deliberately NOT this card's current box. The endpoint intersects the
      // products' footprints with whatever `aoi` it is given, so passing the
      // card's own box made the answer "your box, again" — the button then
      // looked broken on a filled-in card and only appeared to work after
      // "clear". This asks the question the button actually poses: what is the
      // overlap of the two selected maps?
      aoi: null,
      sample_scale_m: Number($("sample_scale_m").value) || null,
    };
    const r = await getJSON("/api/overlap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const bbox = r.bbox.map((x) => x.toFixed(3)).join(", ");

    if (r.blocker) {
      info.className = "info error";
      info.textContent = "✖ " + r.blocker;
      return;
    }

    if (r.is_global) {
      // Nothing to adopt: leave the card blank, which already means "the maps'
      // full overlap" for the primary card.
      info.textContent =
        `“${cardLabel(card)}” overlap: GLOBAL (both maps global). ` +
        `bbox [${bbox}]. Draw or type an AOI to bound the run.`;
      return;
    }

    // Adopt it as the card's AOI.
    const applied = r.bbox.map((x) => Number(x.toFixed(5)));
    setCardBbox(card, applied);
    setActiveCard(card.key);
    drawAoiRect(applied);
    await refreshFootprints();

    let msg =
      `“${cardLabel(card)}” AOI set to the maps' overlap: [${bbox}] ` +
      `(bounding box — areas inside it with no data in either map are skipped ` +
      `when sampling).`;

    // Adopting the full overlap at the current scale can be a 3-hour run: the
    // default 10 m over this overlap is ~27 billion pixels PER MAP. The whole
    // point of computing `suggested_scale_m` is to avoid that, so apply it here
    // rather than only mentioning it — a suggestion the user has to notice, do
    // arithmetic on, and hand-enter is one they will miss, and the failure mode
    // is a run that looks hung for hours.
    //
    // Only ever coarsens. If the user has already chosen a scale coarser than
    // the suggestion, theirs is kept: they have accepted a cheaper run, and
    // silently making it finer would be the same surprise in the other
    // direction.
    if (r.estimated_seconds != null) {
      const mins = (s) => Math.max(1, Math.round(s / 60));
      const scaleEl = $("sample_scale_m");
      const current = Number(scaleEl.value) || r.sample_scale_m;
      if (r.suggested_scale_m != null && r.suggested_scale_m > current) {
        scaleEl.value = r.suggested_scale_m;
        msg +=
          `  Sample scale raised ${current} m → ${r.suggested_scale_m} m ` +
          `(≈ ${mins(r.suggested_estimated_seconds)} min for both maps; ` +
          `${current} m would be ≈ ${mins(r.estimated_seconds)} min). ` +
          `Lower it again if you want the finer detail.`;
      } else {
        msg += `  Sampling both maps at ${current} m ≈ ${mins(r.estimated_seconds)} min.`;
      }
    }
    if (r.slow_warning) {
      info.className = "info warn";
      info.textContent = msg + "  ⚠ " + r.slow_warning;
    } else {
      info.textContent = msg;
    }
  } catch (e) {
    info.className = "info error";
    info.textContent = "No overlap / error: " + e.message;
  }
}

// --- AOI absence + auxiliary cards (Stage 7b/7c) -----------------------------

// The AOI list is server-side truth (cache/aois.json): the absence report and
// the merged-table payloads both carry the active auxiliaries, so the UI never
// keeps its own copy of which AOIs exist. There is no post-run absence popup:
// what is absent (and from which AOI) lives in the ⓘ coverage dialog and the
// card list's footer, both refreshed after every run.

// A legend swatch, from the Harmonize legends. Class colours are legend DATA,
// and the API returns them as bare 6-digit hex (Earth Engine palette format),
// so the "#" is added here — same as swatchFor() does for Review.
function swatchForLegend(side, value) {
  const sw = document.createElement("span");
  sw.className = "swatch";
  const c = (LEGENDS[side] || []).find((x) => x.value === value);
  if (c && c.color) sw.style.background = "#" + c.color;
  return sw;
}

// "＋ add AOI" / the coverage dialog's "Add another AOI…" (Stage 7c): append a
// draft auxiliary card. The user draws/types/uploads a box ON THE CARD, then
// clicks its Sample button. The primary run's result stands — its GMMs and
// crosswalk are valid regardless; an auxiliary tops it up with extra rows,
// targeted at the still-absent classes.
function addAoiCard() {
  closeCoverageModal();
  const card = {
    key: "aux" + ++AOI_SEQ,
    isPrimary: false,
    name: "",
    bbox: ["", "", "", ""],
    sampled: false,
    entry: null,
    expanded: true,
  };
  AOIS.push(card);
  renderAoiCards();
  setActiveCard(card.key);
  if (card.el) card.el.scrollIntoView({ block: "nearest" });
  const info = $("overlap-info");
  info.className = "info";
  info.textContent =
    "Draw, upload, or type a bounding box on the new card (e.g. a coast for " +
    "Mangroves), then click its “Sample” button. Only the classes no other AOI " +
    "has modelled (and the other map's co-present classes) are sampled there — " +
    "the primary AOI is not re-sampled.";
}

// Sample the PRIMARY card's points (Stage 2 only, mode="sample"): collect and
// cache the points; no GMMs, no crosswalk — "Run all AOIs" does that from the
// caches. Same function shape as the auxiliary cards' Sample button.
async function samplePrimaryCard(card) {
  let aoi;
  try {
    aoi = cardBbox(card);
  } catch (e) {
    setCardProg(card, e.message, true);
    return false;
  }
  setRunButtons(true);
  try {
    const body = {
      ...selectedPair(),
      aoi,
      mode: "sample",
      sample_scale_m: Number($("sample_scale_m").value) || null,
      n_components: Number($("n_components").value) || null,
      points_floor: Number($("points_floor").value) || null,
      points_target: Number($("points_target").value) || null,
      force_refresh: $("force_refresh").checked,
    };
    setCardProg(card, "Launching point sampling…");
    const { job_id } = await getJSON("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    while (true) {
      const s = await getJSON(`/api/jobs/${job_id}`);
      const pct = Math.round((s.progress || 0) * 100);
      setCardProg(card, `${s.state} — ${s.stage} (${pct}%)`);
      if (s.state === "done") {
        const a = s.aux || {};
        const nRef = ((a.reference || {}).sampled_ok || []).length;
        const nCmp = ((a.compare || {}).sampled_ok || []).length;
        card.dirty = false;
        setCardProg(
          card,
          `points sampled — ${nRef} + ${nCmp} classes (reference + compare)` +
            (a.reused ? " (reused cache)" : "") +
            " — Run all AOIs to fit the models"
        );
        await refreshAoiList();
        return true;
      }
      if (s.state === "failed") {
        setCardProg(card, "Sampling failed: " + (s.error || "unknown error"), true);
        return false;
      }
      await sleep(1500);
    }
  } catch (e) {
    setCardProg(card, "Error: " + e.message, true);
    return false;
  } finally {
    setRunButtons(false);
  }
}

// Sample one auxiliary card. Requires a sampled primary (the server enforces
// it too). ``opts.fit`` fits the GMMs as well (Run all); the card button
// collects points only. On success the draft becomes server truth.
async function sampleCard(card, opts) {
  let aoi;
  try {
    aoi = cardBbox(card);
  } catch (e) {
    setCardProg(card, e.message, true);
    return false;
  }
  if (!aoi) {
    setCardProg(card, "Enter or draw a bounding box for this auxiliary AOI first.", true);
    return false;
  }
  // Cached points with a matching box + parameters are reused server-side;
  // only the shared "force refresh" checkbox forces a fresh GEE sample.
  const force = $("force_refresh").checked;
  setRunButtons(true);
  try {
    const ok = await runAuxJob(
      {
        ...selectedPair(),
        aoi,
        name: card.name && card.name.trim() ? card.name.trim() : null,
        sample_scale_m: Number($("sample_scale_m").value) || null,
        n_components: Number($("n_components").value) || null,
        points_floor: Number($("points_floor").value) || null,
        points_target: Number($("points_target").value) || null,
        force_refresh: force,
        target_side: card.targetSide || "both",
        fit_models: !!(opts && opts.fit),
      },
      card
    );
    if (ok) await refreshMergedTable();
    return ok;
  } finally {
    setRunButtons(false);
  }
}

// Launch one auxiliary-AOI job and poll it to completion, reporting into the
// card's progress line. Returns true on success. On done the card takes the
// server's (possibly auto-generated) name so the refresh absorbs the draft.
async function runAuxJob(body, card) {
  try {
    setCardProg(card, "Launching auxiliary sampling…");
    const { job_id } = await getJSON("/api/aoi/auxiliary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    while (true) {
      const s = await getJSON(`/api/jobs/${job_id}`);
      const pct = Math.round((s.progress || 0) * 100);
      setCardProg(card, `${s.state} — ${s.stage} (${pct}%)`);
      if (s.state === "done") {
        const a = s.aux || {};
        const modelled = []
          .concat((a.reference || {}).modelled || [], (a.compare || {}).modelled || []);
        const sampled = []
          .concat(
            (a.reference || {}).found || [], (a.reference || {}).co_present || [],
            (a.compare || {}).found || [], (a.compare || {}).co_present || []
          );
        if (a.name) card.name = a.name;
        card.dirty = false;
        card.tsTouched = false;
        setCardProg(
          card,
          (body.fit_models
            ? `done — ${modelled.length} class(es) modelled`
            : `points sampled — ${sampled.length} class(es) — Run all AOIs to fit the models`) +
            (a.reused ? " (reused cache)" : "")
        );
        await refreshAoiList();
        return true;
      }
      if (s.state === "failed") {
        setCardProg(card, "Sampling failed: " + (s.error || "unknown error"), true);
        return false;
      }
      await sleep(1500);
    }
  } catch (e) {
    setCardProg(card, "Error: " + e.message, true);
    return false;
  }
}

// Fetch the merged (multi-AOI) matching table and swap it into the current
// results view: rows tagged with evidence_aoi, absence net of auxiliaries.
async function refreshMergedTable() {
  if (!RESULTS) return;
  const q = new URLSearchParams({
    reference_id: RESULTS.reference_id,
    compare_id: RESULTS.compare_id,
  });
  const m = await getJSON(`/api/merged/table?${q}`);
  RESULTS.matching_table = m.matching_table;
  RESULTS.absence = m.absence;
  RESULTS.auxiliaries = m.auxiliaries;
  RESULTS.max_auxiliary_aois = m.max_auxiliary_aois;
  drawSankey(RESULTS);
  renderCsvView();
}

// --- AOI manager (Stage 7c) --------------------------------------------------
// One entry per AOI of the run — the primary plus each auxiliary — listing the
// classes that AOI evidences per side, with rename / re-sample / delete on the
// auxiliaries. The list is server truth (cache/aois.json + the GMM caches).

async function refreshAoiList() {
  let d = null;
  try {
    const q = new URLSearchParams(selectedPair());
    d = await getJSON(`/api/aoi/list?${q}`);
  } catch (e) {
    $("map-status").textContent = `AOI list unavailable: ${e.message}`;
  }
  if (d) mergeServerAois(d);
  renderAoiCards();
  // Keep the ⓘ coverage dialog live if it is open (e.g. a run just finished).
  if (!$("coverage-modal").hidden) renderCoverageLists();
  // Keep the Review maps' AOI rectangles (primary + auxiliaries) in step too.
  if (REV.maps.ref) drawReviewAoi(readAoiSafe());
}

// Fold the server's AOI inventory into the cards, preserving card object
// identity (a polling job keeps writing into its card across re-renders) and
// any local drafts. Server bbox wins unless the user has edited the card since
// (dirty), so a re-run with a new box is never silently reverted.
function mergeServerAois(d) {
  SERVER_AOIS = d;
  const asStrings = (bbox) => (bbox ? bbox.map(String) : ["", "", "", ""]);

  let p = primaryCard();
  if (!p) {
    // Expanded when there is no primary run yet: the box inputs are the first
    // thing a new user needs.
    p = { key: "primary", isPrimary: true, name: "primary", bbox: ["", "", "", ""], sampled: false, entry: null, expanded: !d.primary };
  }
  if (d.primary) {
    p.entry = d.primary;
    p.sampled = true;
    if (!p.dirty) p.bbox = asStrings(d.primary.bbox);
  }

  const next = [p];
  for (const a of d.auxiliaries) {
    const existing = AOIS.find((c) => !c.isPrimary && c.name === a.name);
    if (existing) {
      existing.sampled = true;
      existing.entry = a;
      existing.disabled = !!a.disabled;
      if (!existing.dirty) existing.bbox = asStrings(a.bbox);
      // Keep an unsaved target-side change (touched but not yet re-sampled).
      if (!existing.tsTouched) existing.targetSide = a.target_side || "both";
      next.push(existing);
    } else {
      next.push({
        key: "aux" + ++AOI_SEQ,
        isPrimary: false,
        name: a.name,
        bbox: asStrings(a.bbox),
        sampled: true,
        entry: a,
        targetSide: a.target_side || "both",
        disabled: !!a.disabled,
      });
    }
  }
  // Local drafts (not yet sampled) stay where they were.
  for (const c of AOIS) {
    if (!c.isPrimary && !c.sampled && !next.includes(c)) next.push(c);
  }
  AOIS = next;
}

// Rebuild the card list DOM from AOIS (+ the coverage footer from server truth).
function renderAoiCards() {
  const host = $("aoi-list");
  if (!primaryCard()) {
    AOIS.unshift({ key: "primary", isPrimary: true, name: "primary", bbox: ["", "", "", ""], sampled: false, entry: null, expanded: true });
  }
  if (!AOIS.find((c) => c.key === ACTIVE_AOI)) ACTIVE_AOI = "primary";
  host.innerHTML = "";
  for (const card of AOIS) host.appendChild(aoiCardEl(card));

  const d = SERVER_AOIS;
  const foot = document.createElement("p");
  foot.className = "note";
  if (!d || !d.primary) {
    foot.textContent =
      "No primary run yet: set the primary card's box (or leave it blank for the " +
      "maps' full overlap) and run it. Auxiliary AOIs unlock after that.";
  } else {
    const missing = [].concat(
      d.still_absent.reference.map((c) => `${c.class_name} (${productName(selectedPair().reference_id)})`),
      d.still_absent.compare.map((c) => `${c.class_name} (${productName(selectedPair().compare_id)})`)
    );
    foot.textContent = missing.length
      ? `Not covered by any AOI yet: ${missing.join(", ")}. ` +
        `${d.auxiliaries.length}/${d.max_auxiliary_aois} auxiliaries used.`
      : `Every declared class of both maps is covered. ` +
        `${d.auxiliaries.length}/${d.max_auxiliary_aois} auxiliaries used.`;
  }
  host.appendChild(foot);
  $("aoi-add").disabled = !!(d && d.auxiliaries.length >= d.max_auxiliary_aois);
}

// Switch the ACTIVE card (no DOM rebuild — a rebuild would steal input focus):
// retint the cards and mirror the card's box onto the maps.
function setActiveCard(key) {
  if (ACTIVE_AOI === key) return;
  ACTIVE_AOI = key;
  for (const c of AOIS) if (c.el) c.el.classList.toggle("active", c.key === key);
  drawAoiRect(activeBboxSafe());
  refreshFootprints();
}

// --- Class-coverage dialog (the ⓘ button) -----------------------------------
// Which AOI evidences each class, per side, plus what is still absent on each
// map — the same server truth the card list uses, shown on demand. This is
// where absence is reported (there is no post-run popup): add another AOI for
// the absent classes, or continue.

async function showCoverageModal() {
  try {
    const q = new URLSearchParams(selectedPair());
    mergeServerAois(await getJSON(`/api/aoi/list?${q}`));
    renderAoiCards();
  } catch (_) {
    // fall back to the last known payload (if any)
  }
  renderCoverageLists();
  $("coverage-modal").hidden = false;
}

// The dialog's filters survive re-renders (the dialog live-updates after runs).
const COV_FILTER = { map: "all", aoi: "all", status: "all" };

// One row per (map, class): which AOI(s) evidence it, or absent. A class can be
// evidenced by more than one AOI (an auxiliary also samples the other map's
// co-present classes), so the AOI cell may list several names.
function coverageRows(d) {
  const rows = [];
  for (const [side, key] of [["ref", "reference"], ["cmp", "compare"]]) {
    const map = productName(selectedPair()[`${key}_id`]).split(" ")[0];
    const byClass = new Map(); // value -> {name, aois}
    const add = (c, aoi) => {
      const e = byClass.get(c.value) || { name: c.name, aois: [] };
      e.aois.push(aoi);
      byClass.set(c.value, e);
    };
    if (d.primary) for (const c of (d.primary[key] || {}).modelled || []) add(c, "primary");
    // Unused auxiliaries are set aside: their classes do not count as covered.
    for (const a of d.auxiliaries) {
      if (a.disabled) continue;
      for (const c of (a[key] || {}).modelled || []) add(c, a.name);
    }
    for (const [value, e] of byClass) {
      rows.push({ side, map, value, name: e.name, aois: e.aois, absent: false });
    }
    for (const c of d.still_absent[key] || []) {
      rows.push({
        side, map,
        value: c.class_value,
        name: c.class_name,
        aois: [],
        absent: true,
        reason: c.reason,
      });
    }
    rows.sort((a, b) => (a.side === b.side ? a.value - b.value : 0));
  }
  return rows;
}

// Rebuild the dialog's content from SERVER_AOIS: a filter bar (by AOI, by
// present/absent) + one flat table, one class per row. Also called by
// refreshAoiList while the dialog is open, so a run finishing updates it live.
function renderCoverageLists() {
  const host = $("coverage-lists");
  host.innerHTML = "";
  const d = SERVER_AOIS;
  const addBtn = $("coverage-add-aoi");
  if (!d || !d.primary) {
    const p = document.createElement("p");
    p.className = "note";
    p.textContent =
      "No run yet — coverage is known only after sampling. Run the primary AOI first.";
    host.appendChild(p);
    addBtn.disabled = true;
    addBtn.title = "Run the primary AOI first — an auxiliary tops up a primary run.";
    return;
  }

  // Filter bar. Selects rebuild each render; their values come from COV_FILTER.
  // Unused auxiliaries are excluded — they contribute no coverage.
  const aoiNames = [
    "primary",
    ...d.auxiliaries.filter((a) => !a.disabled).map((a) => a.name),
  ];
  if (!["all", ...aoiNames].includes(COV_FILTER.aoi)) COV_FILTER.aoi = "all";
  const bar = document.createElement("div");
  bar.className = "row coverage-filters";
  const mkSelect = (labelText, options, value, onChange) => {
    const label = document.createElement("label");
    label.className = "inline";
    label.appendChild(document.createTextNode(labelText));
    const sel = document.createElement("select");
    for (const [v, text] of options) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = text;
      sel.appendChild(o);
    }
    sel.value = value;
    sel.addEventListener("change", () => onChange(sel.value));
    label.appendChild(sel);
    return label;
  };
  bar.appendChild(
    mkSelect(
      "map",
      [
        ["all", "both maps"],
        ["ref", productName(selectedPair().reference_id).split(" ")[0]],
        ["cmp", productName(selectedPair().compare_id).split(" ")[0]],
      ],
      COV_FILTER.map,
      (v) => {
        COV_FILTER.map = v;
        renderCoverageLists();
      }
    )
  );
  bar.appendChild(
    mkSelect(
      "AOI",
      [["all", "all AOIs"], ...aoiNames.map((n) => [n, n])],
      COV_FILTER.aoi,
      (v) => {
        COV_FILTER.aoi = v;
        renderCoverageLists();
      }
    )
  );
  bar.appendChild(
    mkSelect(
      "status",
      [["all", "present + absent"], ["present", "present"], ["absent", "absent"]],
      COV_FILTER.status,
      (v) => {
        COV_FILTER.status = v;
        renderCoverageLists();
      }
    )
  );
  host.appendChild(bar);

  const rows = coverageRows(d).filter((r) => {
    if (COV_FILTER.map !== "all" && r.side !== COV_FILTER.map) return false;
    if (COV_FILTER.status === "present" && r.absent) return false;
    if (COV_FILTER.status === "absent" && !r.absent) return false;
    if (COV_FILTER.aoi !== "all" && !r.aois.includes(COV_FILTER.aoi)) return false;
    return true;
  });

  const table = document.createElement("table");
  table.className = "coverage-table";
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const h of ["map", "class", "AOI", "status"]) {
    const th = document.createElement("th");
    th.textContent = h;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  for (const r of rows) {
    const tr = document.createElement("tr");
    const tdMap = document.createElement("td");
    tdMap.className = "cov-map";
    tdMap.textContent = r.map;
    tr.appendChild(tdMap);
    const tdClass = document.createElement("td");
    tdClass.appendChild(swatchForLegend(r.side, r.value));
    tdClass.appendChild(document.createTextNode(` ${r.name} (${r.value})`));
    tr.appendChild(tdClass);
    const tdAoi = document.createElement("td");
    tdAoi.textContent = r.aois.length ? r.aois.join(", ") : "—";
    tr.appendChild(tdAoi);
    const tdStatus = document.createElement("td");
    tdStatus.className = "cov-status " + (r.absent ? "absent" : "present");
    tdStatus.textContent = r.absent
      ? "absent" + (r.reason === "too_rare" ? " (too rare)" : "")
      : "present";
    tdStatus.title = r.absent
      ? "No AOI has modelled this class yet" +
        (r.reason === "too_rare" ? " — it was found, but with too few usable points." : ".")
      : "A GMM is fitted for this class from the listed AOI(s).";
    tr.appendChild(tdStatus);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  if (!rows.length) {
    const p = document.createElement("p");
    p.className = "note";
    p.textContent = "No classes match the current filters.";
    host.appendChild(p);
  } else {
    host.appendChild(table);
  }

  addBtn.disabled = d.auxiliaries.length >= d.max_auxiliary_aois;
  addBtn.title = addBtn.disabled
    ? `Auxiliary AOI limit reached (${d.max_auxiliary_aois} per run).`
    : "Append a draft AOI card; it will sample only the still-absent classes.";
}

function closeCoverageModal() {
  $("coverage-modal").hidden = true;
}

// One AOI card: a SIMPLE collapsed row (arrow + name + role tag + actions);
// the arrow (or the head) expands it to the card's own bbox inputs, tools
// (check overlap / upload / clear), and Run/Sample button + progress. Which
// classes each AOI evidences lives in the ⓘ coverage dialog, not on the card.
function aoiCardEl(card) {
  const row = document.createElement("div");
  row.className =
    "aoi-entry aoi-card" +
    (card.isPrimary ? " aoi-primary" : "") +
    (card.disabled ? " aoi-unused" : "") +
    (card.key === ACTIVE_AOI ? " active" : "");
  card.el = row;
  row.addEventListener("click", () => setActiveCard(card.key));

  // Head: expand arrow + name (fixed for primary and sampled auxes; editable
  // for drafts) + tag. Clicking the head (not its buttons/inputs) toggles.
  const head = document.createElement("div");
  head.className = "aoi-entry-head";
  head.addEventListener("click", (e) => {
    if (e.target.closest("button, input, label")) return;
    toggleCardExpand(card);
  });
  const arrow = document.createElement("button");
  arrow.type = "button";
  arrow.className = "aoi-expand";
  arrow.textContent = card.expanded ? "▾" : "▸";
  arrow.title = "Show this AOI's bounding box and run controls";
  arrow.addEventListener("click", () => toggleCardExpand(card));
  card.arrowEl = arrow;
  const left = document.createElement("span");
  left.className = "aoi-head-left";
  left.appendChild(arrow);
  const title = document.createElement("span");
  title.className = "aoi-entry-name";
  if (card.renaming) {
    // Inline rename editor (native prompt() is a silent no-op in embedded
    // browsers, so renaming happens in place): input + save / cancel.
    const nameInp = document.createElement("input");
    nameInp.type = "text";
    nameInp.className = "aoi-name-input";
    nameInp.value = card.name || "";
    const doRename = async () => {
      const name = nameInp.value.trim();
      if (!name || name === card.name) {
        card.renaming = false;
        renderAoiCards();
        return;
      }
      try {
        await getJSON("/api/aoi/auxiliary/rename", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...selectedPair(), old_name: card.name, new_name: name }),
        });
        card.name = name; // keep identity through the refresh's name match
        card.renaming = false;
        await refreshAoiList();
        await refreshMergedTable(false);
      } catch (err) {
        card.renaming = false;
        renderAoiCards();
        setCardProg(card, "Rename failed: " + err.message, true);
      }
    };
    nameInp.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doRename();
      if (e.key === "Escape") {
        card.renaming = false;
        renderAoiCards();
      }
    });
    title.appendChild(nameInp);
    title.appendChild(miniBtn("save", "Save the new name (no re-sample)", doRename));
    title.appendChild(
      miniBtn("cancel", "Keep the old name", () => {
        card.renaming = false;
        renderAoiCards();
      })
    );
    setTimeout(() => nameInp.focus(), 0);
  } else if (card.isPrimary || card.sampled) {
    title.textContent = cardLabel(card);
  } else {
    const nameInp = document.createElement("input");
    nameInp.type = "text";
    nameInp.className = "aoi-name-input";
    nameInp.placeholder = "name (optional), e.g. mangrove-coast";
    nameInp.value = card.name || "";
    nameInp.addEventListener("input", () => (card.name = nameInp.value));
    title.appendChild(nameInp);
  }
  const tag = document.createElement("span");
  tag.className = "aoi-tag";
  tag.textContent = card.isPrimary
    ? card.sampled ? "primary" : "primary · not run yet"
    : card.disabled ? "auxiliary · unused"
    : card.sampled ? "auxiliary" : "auxiliary · draft";
  tag.title = card.isPrimary
    ? "The main harmonization AOI: its run fits the GMMs and computes the crosswalk."
    : "Auxiliary AOI: tops up the primary run with the classes it could not model.";
  title.appendChild(tag);
  left.appendChild(title);
  head.appendChild(left);

  const btns = document.createElement("span");
  btns.className = "aoi-entry-actions";
  if (!card.isPrimary && card.sampled && !card.renaming) {
    // Unuse / use: the middle ground between keeping and deleting. An unused
    // AOI keeps its cached points/fits on disk but the model ignores it
    // everywhere until it is used again.
    btns.appendChild(
      miniBtn(
        card.disabled ? "use" : "unuse",
        card.disabled
          ? "Use this AOI again — its cached data feeds the model once more (no re-sample)."
          : "Set this AOI aside: keep its cached data but exclude it from the model. Re-usable later.",
        async () => {
          try {
            await getJSON("/api/aoi/auxiliary/use", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                ...selectedPair(),
                name: card.name,
                use: !!card.disabled,
              }),
            });
            await refreshAoiList();
            await refreshMergedTable();
          } catch (err) {
            setCardProg(card, "Use toggle failed: " + err.message, true);
          }
        }
      )
    );
    btns.appendChild(
      miniBtn("rename", "Rename this auxiliary in place (no re-sample)", () => {
        card.renaming = true;
        renderAoiCards();
      })
    );
    // Two-step delete (native confirm() is a silent no-op in embedded
    // browsers): first click arms the button, second click deletes.
    btns.appendChild(
      miniBtn("delete", "Remove this auxiliary and its caches — click twice to confirm", async (btn) => {
        if (!btn.dataset.armed) {
          btn.dataset.armed = "1";
          btn.textContent = "confirm delete?";
          btn.classList.add("danger");
          setTimeout(() => {
            if (!btn.isConnected) return;
            delete btn.dataset.armed;
            btn.textContent = "delete";
            btn.classList.remove("danger");
          }, 4000);
          return;
        }
        try {
          const q = new URLSearchParams(selectedPair());
          await getJSON(`/api/aoi/auxiliary/${encodeURIComponent(card.name)}?${q}`, {
            method: "DELETE",
          });
          AOIS = AOIS.filter((c) => c !== card);
          await refreshAoiList();
          await refreshMergedTable(false);
        } catch (err) {
          setCardProg(card, "Delete failed: " + err.message, true);
        }
      })
    );
  }
  if (!card.isPrimary && !card.sampled) {
    btns.appendChild(
      miniBtn("remove", "Discard this draft card (nothing was sampled)", () => {
        AOIS = AOIS.filter((c) => c !== card);
        renderAoiCards();
        drawAoiRect(activeBboxSafe());
        refreshFootprints();
      })
    );
  }
  head.appendChild(btns);
  row.appendChild(head);

  // Collapsible body: everything below the name row hides until expanded. The
  // inputs stay live while hidden (map draws still write into them).
  const body = document.createElement("div");
  body.className = "aoi-card-body" + (card.expanded ? "" : " collapsed");
  card.bodyEl = body;

  // The card's own bbox inputs. Blank on the primary = the maps' full overlap.
  const bb = document.createElement("div");
  bb.className = "aoi-bbox";
  card.inputs = [];
  ["min lon", "min lat", "max lon", "max lat"].forEach((ph, i) => {
    const inp = document.createElement("input");
    inp.type = "number";
    inp.step = "any";
    inp.placeholder = ph;
    inp.title = ph;
    inp.value = card.bbox ? card.bbox[i] : "";
    inp.addEventListener("change", () => {
      card.bbox[i] = inp.value;
      card.dirty = true;
      if (card.key === ACTIVE_AOI) {
        drawAoiRect(activeBboxSafe());
        refreshFootprints();
        if (card.isPrimary && REV.maps.ref) fitReviewToAoi();
      }
    });
    card.inputs.push(inp);
    bb.appendChild(inp);
  });
  body.appendChild(bb);

  // Auxiliary only: WHOSE absent classes this AOI targets. An AOI picked for
  // one map's missing class (a coast for Mangroves) should not also chase the
  // other map's unrelated absences there — the other map's co-present classes
  // are sampled regardless (they make the edges within-AOI comparisons).
  if (!card.isPrimary) {
    const tsLabel = document.createElement("label");
    tsLabel.className = "inline aoi-target";
    tsLabel.title =
      "Which map's still-absent classes this AOI samples. The other map's " +
      "co-present classes are sampled here regardless — that is what makes " +
      "the comparison valid. Changing this re-samples on the next Sample.";
    tsLabel.appendChild(document.createTextNode("target absent classes of"));
    const tsSel = document.createElement("select");
    const refShort = productName(selectedPair().reference_id).split(" ")[0];
    const cmpShort = productName(selectedPair().compare_id).split(" ")[0];
    for (const [v, text] of [
      ["both", "both maps"],
      ["reference", `${refShort} only`],
      ["compare", `${cmpShort} only`],
    ]) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = text;
      tsSel.appendChild(o);
    }
    tsSel.value = card.targetSide || "both";
    tsSel.addEventListener("change", () => {
      card.targetSide = tsSel.value;
      card.tsTouched = true;
    });
    tsLabel.appendChild(tsSel);
    body.appendChild(tsLabel);
  }

  // The card's own AOI tools + its own run button.
  const tools = document.createElement("div");
  tools.className = "row aoi-tools";
  // The SAME button on every card (primary and auxiliary alike): make sure
  // this AOI's sample points exist — reused from the cache when the box and
  // parameters are unchanged, sampled from GEE only when they are not (tick
  // "force refresh" to re-sample regardless). No model fitting here — "Run
  // all AOIs" fits the GMMs and computes the crosswalk from the cached points.
  const runBtn = document.createElement("button");
  runBtn.type = "button";
  runBtn.className = "primary card-run";
  runBtn.textContent = "Sample points";
  if (card.isPrimary) {
    runBtn.title =
      "Ensure this AOI's sample points are cached: reused when box + parameters " +
      "are unchanged, sampled from GEE otherwise (points only — “Run all AOIs” " +
      "fits the models and computes the crosswalk).";
    runBtn.addEventListener("click", () => samplePrimaryCard(card));
  } else {
    runBtn.title =
      "Ensure this AOI's sample points are cached for the targeted classes: " +
      "reused when box + parameters are unchanged, sampled from GEE otherwise " +
      "(points only — “Run all AOIs” fits the models and computes the crosswalk).";
    if (!(SERVER_AOIS && SERVER_AOIS.primary)) {
      runBtn.disabled = true;
      runBtn.title = "Sample the primary AOI first — an auxiliary tops up a primary run.";
    }
    runBtn.addEventListener("click", () => sampleCard(card));
  }
  card.runBtn = runBtn;
  tools.appendChild(runBtn);
  tools.appendChild(
    miniBtn(
      "use overlap",
      "Compute the two products' overlap and set it as this AOI",
      () => checkOverlap(card)
    )
  );
  const up = document.createElement("label");
  up.className = "mini-upload";
  up.title = "Set this AOI from a GeoJSON boundary's bounding box";
  up.textContent = "upload";
  const file = document.createElement("input");
  file.type = "file";
  file.accept = ".json,.geojson,application/json,application/geo+json";
  file.hidden = true;
  file.addEventListener("change", () => {
    if (file.files && file.files[0]) uploadAoi(file.files[0], card);
    file.value = "";
  });
  up.appendChild(file);
  tools.appendChild(up);
  tools.appendChild(
    miniBtn("clear", "Blank this AOI's box (primary: use the maps' full overlap)", () => {
      setCardBbox(card, null);
      if (card.key === ACTIVE_AOI) {
        drawAoiRect(null);
        refreshFootprints();
      }
    })
  );
  const prog = document.createElement("span");
  prog.className = "progress" + (card.progErr ? " error" : "");
  prog.textContent = card.progMsg || "";
  card.progEl = prog;
  tools.appendChild(prog);
  body.appendChild(tools);

  // Compact coverage summary — the full per-class breakdown lives in the ⓘ
  // coverage dialog, keeping the card itself simple.
  if (card.entry) {
    const nRef = ((card.entry.reference || {}).modelled || []).length;
    const nCmp = ((card.entry.compare || {}).modelled || []).length;
    const sum = document.createElement("p");
    sum.className = "note aoi-summary";
    sum.textContent = `evidences ${nRef} + ${nCmp} classes (reference + compare) — see ⓘ for the breakdown`;
    body.appendChild(sum);
  }

  row.appendChild(body);
  return row;
}

function toggleCardExpand(card) {
  card.expanded = !card.expanded;
  if (card.bodyEl) card.bodyEl.classList.toggle("collapsed", !card.expanded);
  if (card.arrowEl) card.arrowEl.textContent = card.expanded ? "▾" : "▸";
}

function miniBtn(label, title, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "mini";
  b.textContent = label;
  b.title = title;
  b.addEventListener("click", () => onClick(b));
  return b;
}

// --- Run + poll ------------------------------------------------------------

// Disable/enable every run control at once (the per-card buttons + Run all),
// so only one job talks to GEE at a time. Re-enabling re-applies the
// "auxiliaries need a primary run first" gate.
function setRunButtons(disabled) {
  $("run").disabled = disabled;
  const havePrimary = !!(SERVER_AOIS && SERVER_AOIS.primary);
  for (const c of AOIS) {
    if (!c.runBtn) continue;
    c.runBtn.disabled = disabled || (!c.isPrimary && !havePrimary);
  }
}

// Run the full harmonization on the PRIMARY card's AOI (sampling, GMM fits,
// affinity, crosswalk). Returns true on success — Run all sequences on it.
async function runPrimary() {
  const card = primaryCard();
  let aoi;
  try {
    aoi = cardBbox(card);
  } catch (e) {
    setCardProg(card, e.message, true);
    return false;
  }
  setRunButtons(true);
  try {
    const body = {
      ...selectedPair(),
      aoi,
      sample_scale_m: Number($("sample_scale_m").value) || null,
      n_components: Number($("n_components").value) || null,
      points_floor: Number($("points_floor").value) || null,
      points_target: Number($("points_target").value) || null,
      force_refresh: $("force_refresh").checked,
    };
    setCardProg(card, "Launching…");
    const { job_id } = await getJSON("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    CURRENT_JOB = job_id;
    while (true) {
      const s = await getJSON(`/api/jobs/${job_id}`);
      const pct = Math.round((s.progress || 0) * 100);
      setCardProg(card, `${s.state} — ${s.stage} (${pct}%)`);
      if (s.state === "done") {
        card.dirty = false;
        await showResults(job_id);
        return true;
      }
      if (s.state === "failed") {
        setCardProg(card, "Run failed: " + (s.error || "unknown error"), true);
        return false;
      }
      await sleep(1500);
    }
  } catch (e) {
    setCardProg(card, "Error: " + e.message, true);
    return false;
  } finally {
    setRunButtons(false);
  }
}

// "Run all AOIs": the primary run first (auxiliaries top up a primary run, so
// it must exist), then every auxiliary card in list order. Cached AOIs are
// reused unless "force refresh" is ticked, so re-running the lot is cheap.
async function runAll() {
  const gp = $("progress");
  RUN_ALL = true;
  gp.className = "progress";
  try {
    gp.textContent = "Running primary AOI…";
    const ok = await runPrimary();
    if (!ok) {
      gp.className = "progress error";
      gp.textContent = "Stopped: the primary run failed (auxiliaries need it).";
      return;
    }
    const auxes = AOIS.filter((c) => !c.isPrimary);
    let failed = 0;
    let skipped = 0;
    let unused = 0;
    for (let i = 0; i < auxes.length; i++) {
      const card = auxes[i];
      if (card.disabled) {
        unused++;
        setCardProg(card, "skipped — unused (click “use” to include it)");
        continue;
      }
      if (!cardBboxSafe(card)) {
        skipped++;
        setCardProg(card, "skipped — no bounding box", true);
        continue;
      }
      gp.textContent = `Fitting auxiliary AOI ${i + 1}/${auxes.length} (“${cardLabel(card)}”)…`;
      if (!(await sampleCard(card, { fit: true }))) failed++;
    }
    gp.className = failed ? "progress error" : "progress";
    gp.textContent =
      "Run all finished" +
      (failed ? ` — ${failed} auxiliary AOI(s) failed` : "") +
      (skipped ? ` — ${skipped} draft(s) skipped (no box)` : "") +
      (unused ? ` — ${unused} unused AOI(s) skipped` : "") +
      ". See ⓘ for class coverage.";
  } finally {
    RUN_ALL = false;
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// --- Results: mapping Sankey + result-table viewer -------------------------

let SANKEY = null; // single ECharts instance on #sankey
let RESULTS = null; // last results payload (for the table viewer)

// --- Stage 8d: semantic-prior alpha ---------------------------------------
// Alpha is a DISPLAY control, not a run parameter: it sits outside the run
// signature, so changing it re-decides Stage 4 from the cached models with no
// GEE call and no re-sampling. It is kept in this module variable (session
// state) rather than written back to config.py — exploration must not silently
// rewrite the calibration Stage 8c established. Closing the app restores the
// calibrated default.
let ALPHA = null; // null until the defaults land; then the user's chosen value
let ALPHA_DEFAULT = null; // the calibrated CONFIG value, for the anchor label

function currentAlpha() {
  return ALPHA == null ? (ALPHA_DEFAULT == null ? 1 : ALPHA_DEFAULT) : ALPHA;
}

// Reflect the slider position in its readout and note, without fetching.
function paintAlpha() {
  const a = currentAlpha();
  $("alpha").value = String(a);
  $("alpha-value").textContent = fmt(a, 2);
  const note = $("alpha-note");
  if (ALPHA_DEFAULT == null) {
    note.textContent = "";
  } else if (Math.abs(a - ALPHA_DEFAULT) < 1e-9) {
    note.textContent = `calibrated value (α = ${fmt(ALPHA_DEFAULT, 2)})`;
    note.classList.remove("alpha-off-default");
  } else {
    // Say plainly that the view no longer matches the project's calibration:
    // the CLI scripts still use the config value, so the two can disagree.
    note.textContent =
      `display only — differs from the calibrated α = ${fmt(ALPHA_DEFAULT, 2)}` +
      " used by config and the command-line scripts";
    note.classList.add("alpha-off-default");
  }
}

// Re-decide at the current alpha from the cached models and repaint. Cheap
// (no GEE), but still guarded: it needs a completed run for the pair.
async function applyAlpha() {
  if (!RESULTS) return;
  const wantAef = $("alpha-compare").checked;
  const q = new URLSearchParams({
    reference_id: RESULTS.reference_id,
    compare_id: RESULTS.compare_id,
    alpha: String(currentAlpha()),
    include_aef: wantAef ? "true" : "false",
  });
  $("alpha-row").classList.add("busy");
  try {
    const r = await getJSON(`/api/affinity?${q}`);
    RESULTS.matching_table = r.matching_table;
    RESULTS.normalized_affinity = r.normalized_affinity;
    RESULTS.raw_similarity = r.raw_similarity;
    RESULTS.matching_table_aef = wantAef ? r.matching_table_aef : null;
    drawSankey(RESULTS);
    renderCsvView();
  } catch (e) {
    setCardProg(
      primaryCard(),
      `could not re-decide at α = ${fmt(currentAlpha(), 2)}: ${e.message}`,
      true
    );
  } finally {
    $("alpha-row").classList.remove("busy");
  }
}

// Enable/disable the control: it recomputes from cached models, so before any
// run there is nothing to recompute and the slider must say so rather than
// silently returning an empty table.
function setAlphaEnabled(on) {
  $("alpha").disabled = !on;
  $("alpha-compare").disabled = !on;
  $("alpha-panel").classList.toggle("disabled", !on);
  if (!on) $("alpha-note").textContent = "run a harmonization to enable";
  else paintAlpha();
}

function wireAlpha() {
  // input → live readout (cheap); change → the actual recompute on release, so
  // dragging does not fire a request per step.
  $("alpha").addEventListener("input", () => {
    ALPHA = Number($("alpha").value);
    paintAlpha();
  });
  $("alpha").addEventListener("change", () => {
    ALPHA = Number($("alpha").value);
    paintAlpha();
    applyAlpha();
  });
  $("alpha-compare").addEventListener("change", applyAlpha);
  setAlphaEnabled(false);
}

async function showResults(jobId) {
  const r = await getJSON(`/api/jobs/${jobId}/results`);
  RESULTS = r;
  RESULTS.job_id = jobId;
  setCardProg(
    primaryCard(),
    r.reused_cache
      ? "done — reused cached GMMs (no GEE sampling)"
      : "done — sampled fresh from GEE"
  );
  drawSankey(r);
  renderCsvView();

  // Stage 8d: the run computed at the CONFIG alpha. If this session's slider
  // sits elsewhere, re-decide so the view matches the control rather than
  // silently showing a different alpha than the one displayed.
  setAlphaEnabled(true);
  const runAlpha = r.calibration ? r.calibration.semantic_prior_alpha : null;
  if (
    ALPHA != null &&
    runAlpha != null &&
    Math.abs(ALPHA - runAlpha) > 1e-9
  ) {
    await applyAlpha();
  } else if ($("alpha-compare").checked) {
    await applyAlpha();
  }

  // Stage 7c: if auxiliaries already top up this pair, the deliverable is the
  // MERGED table (union of every AOI's rows, evidence_aoi-tagged), not the
  // primary-only rows this job returned — swap it in.
  if (r.absence && (r.absence.auxiliaries || []).length) {
    await refreshMergedTable();
  }
  // The run (re)recorded the primary AOI; keep the AOI manager (and, if open,
  // the ⓘ coverage dialog) in step. What could not be modelled is reported
  // there and in the card list's footer — no popup.
  refreshAoiList();
}

// Look up the land-cover colour (legend DATA) for a class name on one side, so the
// Sankey nodes reuse the map/legend colours. Returns "#rrggbb" or null.
function classColor(side, name) {
  const legend = LEGENDS[side];
  if (!legend) return null;
  const c = legend.find((x) => x.name === name);
  return c ? "#" + c.color : null;
}

// Sankey of the mapping probabilities: reference classes (left) → compare classes
// (right), link width = probability. Node names are namespaced per side so a class
// name present on both sides does not collapse into one node.
function drawSankey(r) {
  const el = $("sankey");
  if (!SANKEY) SANKEY = echarts.init(el, null, { renderer: "canvas" });

  const nodes = [];
  const seen = new Set();
  // Reference (left) nodes label to the right; compare (right) nodes label to the
  // LEFT (inward) so the right-column labels never clip off the panel edge.
  const addNode = (id, label, color, side) => {
    if (seen.has(id)) return;
    seen.add(id);
    const item = {
      name: id,
      label: { formatter: label, position: side === "cmp" ? "left" : "right" },
    };
    if (color) item.itemStyle = { color };
    nodes.push(item);
  };

  const links = [];
  for (const row of r.matching_table) {
    const refId = "ref:" + row.reference_name;
    addNode(refId, row.reference_name, classColor("ref", row.reference_name), "ref");
    for (const c of row.compare) {
      if (!c.probability || c.probability <= 0.001) continue; // keep it readable
      const cmpId = "cmp:" + c.name;
      addNode(cmpId, c.name, classColor("cmp", c.name), "cmp");
      links.push({ source: refId, target: cmpId, value: c.probability });
    }
  }

  SANKEY.setOption({
    backgroundColor: "transparent",
    tooltip: {
      trigger: "item",
      formatter: (p) =>
        p.dataType === "edge"
          ? `${p.data.source.replace(/^ref:/, "")} → ${p.data.target.replace(/^cmp:/, "")}<br/>p = ${p.data.value.toFixed(3)}`
          : p.name.replace(/^(ref|cmp):/, ""),
    },
    series: [
      {
        type: "sankey",
        left: 12,
        right: 20,
        top: 10,
        bottom: 10,
        nodeGap: 8,
        nodeWidth: 14,
        emphasis: { focus: "adjacency" },
        data: nodes,
        links: links,
        label: { color: "#e6edf3", fontSize: 11 },
        lineStyle: { color: "gradient", opacity: 0.45, curveness: 0.5 },
      },
    ],
  }, true);
  SANKEY.resize();
}

function fmt(x, d = 3) {
  return x === null || x === undefined ? "—" : Number(x).toFixed(d);
}

// Render the selected result table (matching / affinity / similarity) into the CSV
// panel from the last results payload, and point the download link at its export.
function renderCsvView() {
  const view = $("csv-view");
  if (!RESULTS) {
    view.innerHTML = `<p class="note">Run a harmonization to view its result tables.</p>`;
    return;
  }
  const which = $("csv-pick").value;
  // The matching table deliverable is the MERGED table (union of every AOI's
  // rows, evidence_aoi column included — Stage 7.4); the matrices stay per-run.
  // Stage 8d: the matrices carry the current alpha so a download matches what is
  // on screen. The merged table has its own multi-AOI path and no alpha param.
  $("csv-download").href =
    which === "matching_table"
      ? `/api/merged/export?reference_id=${encodeURIComponent(RESULTS.reference_id)}&compare_id=${encodeURIComponent(RESULTS.compare_id)}`
      : `/api/jobs/${RESULTS.job_id}/export/${which}?alpha=${encodeURIComponent(currentAlpha())}`;
  view.innerHTML = "";

  if (which !== "matching_table") {
    view.appendChild(matrixTable(which));
    return;
  }

  // Stage 8d comparison view: when the α=0 table is also loaded, show it
  // beneath the current one, each under its own heading, so the effect of the
  // prior is readable without switching back and forth.
  const aef = RESULTS.matching_table_aef;
  if (aef && Math.abs(currentAlpha()) > 1e-9) {
    const wrap = document.createElement("div");
    wrap.className = "table-compare";
    wrap.appendChild(
      tableHeading(`α = ${fmt(currentAlpha(), 2)} — with semantic prior`)
    );
    wrap.appendChild(matchingTable(RESULTS.matching_table));
    wrap.appendChild(tableHeading("α = 0 — observational only"));
    wrap.appendChild(matchingTable(aef));
    view.appendChild(wrap);
    return;
  }
  view.appendChild(matchingTable(RESULTS.matching_table));
}

// A small heading above one table in the α comparison view.
function tableHeading(text) {
  const h = document.createElement("p");
  h.className = "table-heading";
  h.textContent = text;
  return h;
}

// The Stage-4 matching table, as an on-screen table. ``rows`` defaults to the
// current results so existing callers are unchanged; the α comparison view
// passes the α=0 rows to render a second table from the same code.
function matchingTable(rows) {
  const t = document.createElement("table");
  t.className = "matching";
  t.innerHTML =
    "<thead><tr>" +
    "<th>reference</th><th>status</th><th>maps to (probability)</th>" +
    "<th>evidence AOI</th>" +
    "<th>best raw</th><th>margin</th><th>entropy</th><th>confidence</th>" +
    "</tr></thead>";
  const tb = document.createElement("tbody");
  for (const row of rows || RESULTS.matching_table) {
    const tr = document.createElement("tr");
    tr.className = "status-" + row.status;
    // A compare-side row (Stage 7) reports a compare class nothing could map to,
    // so it names that class and has no candidates of its own.
    const isCompareSide = row.side === "compare";
    const label = isCompareSide
      ? `${row.compare[0] ? row.compare[0].name : ""} (compare)`
      : row.reference_name;
    const candidates = isCompareSide
      ? "unmatched target"
      : row.compare
          .map(
            (c) =>
              `${c.name} (${fmt(c.probability, 2)})` +
              (c.low_confidence ? " ⚠" : "")
          )
          .join(", ");
    // An absent class says why it could not be modelled, so the reader can tell
    // "never occurs here" from "too rare to fit".
    const statusText = row.absence_reason
      ? `${row.status}: ${row.absence_reason}`
      : row.status;
    // Which AOI's evidence produced this row (Stage 7c): "primary", an
    // auxiliary's name, "none" (expert-declared), or "—" (absent, no evidence).
    // Probabilities normalise within one AOI's sub-matrix, so this tag is what
    // tells the reader when two rows are not directly comparable.
    const evidence = row.evidence_aoi || "—";
    tr.innerHTML =
      `<td>${escapeHtml(label)}</td>` +
      `<td><span class="badge ${row.status}">${escapeHtml(statusText)}</span></td>` +
      `<td>${candidates || "—"}</td>` +
      `<td>${escapeHtml(evidence)}</td>` +
      `<td>${fmt(row.best_raw_similarity)}</td>` +
      `<td>${fmt(row.margin)}</td>` +
      `<td>${fmt(row.entropy, 2)}</td>` +
      `<td>${row.reference_low_confidence ? "low ⚠" : "ok"}</td>`;
    // A mixed/orphan row is the entry point to Review (docs/PIPELINE.md 6.7):
    // clicking it enters Review focused on that reference class. Compare-side
    // rows have no reference class to focus, so they are not entry points.
    if (!isCompareSide && (row.status === "mixed" || row.status === "orphan")) {
      tr.classList.add("reviewable");
      tr.title = "Review this class";
      tr.addEventListener("click", () => enterReview(row.reference_value));
    }
    tb.appendChild(tr);
  }
  t.appendChild(tb);
  return t;
}

// A labelled N×M matrix (normalized_affinity or raw_similarity) as a table.
function matrixTable(which) {
  const z = RESULTS[which];
  const cols = RESULTS.compare_labels;
  const rowsLabels = RESULTS.reference_labels;
  const t = document.createElement("table");
  t.className = "matching matrix";
  t.innerHTML =
    "<thead><tr><th>ref \\ cmp</th>" +
    cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("") +
    "</tr></thead>";
  const tb = document.createElement("tbody");
  z.forEach((rowVals, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<th>${escapeHtml(rowsLabels[i])}</th>` +
      rowVals.map((v) => `<td>${fmt(v)}</td>`).join("");
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  return t;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// --- Resizable region dividers ---------------------------------------------

// Recompute the Leaflet map sizes and the Sankey after any layout change.
function syncSizes() {
  if (MAPS.ref) MAPS.ref.invalidateSize();
  if (MAPS.cmp) MAPS.cmp.invalidateSize();
  if (SANKEY) SANKEY.resize();
  if (REV.maps.ref) REV.maps.ref.invalidateSize();
  if (REV.maps.cmp) REV.maps.cmp.invalidateSize();
  if (REV.sankey) REV.sankey.resize();
}

// Drag handles set CSS grid track sizes: the horizontal handle splits the maps row
// vs the bottom band; the two vertical handles size the three bottom columns.
function setupResizers() {
  const main = document.querySelector("main");
  const band = $("control-band");

  // Horizontal: maps-row height (var --maps-height on <main>).
  dragHandle($("resize-rows"), "y", (e) => {
    const rect = main.getBoundingClientRect();
    const h = Math.max(120, Math.min(rect.height - 120, e.clientY - rect.top));
    main.style.setProperty("--maps-height", h + "px");
  });

  // The map|legend divider, one per column. Both write the SAME variable
  // (--legend-height on <main>), so dragging either resizes both legends
  // together and the two maps stay aligned — which is the point: a side-by-side
  // comparison is unreadable if the panes are different heights.
  //
  // Dragging UP grows the legend (useful for a 35-class legend like
  // GLC_FCS30D); dragging DOWN shrinks it back toward one row of chips.
  for (const id of ["resize-legend-ref", "resize-legend-cmp"]) {
    const handle = $(id);
    if (!handle) continue;
    dragHandle(handle, "y", (e) => {
      // Measure against this handle's own column, so the maths is the same
      // whichever side is grabbed.
      const col = handle.closest(".map-col");
      const rect = col.getBoundingClientRect();
      // Distance from the pointer to the bottom of the column = legend height.
      // Floor of 40px keeps one row of chips visible; the cap leaves at least
      // 120px of map so the legend can never swallow it entirely.
      const h = Math.max(40, Math.min(rect.height - 120, rect.bottom - e.clientY));
      main.style.setProperty("--legend-height", Math.round(h) + "px");
    });
  }

  // Vertical: pin the two left column widths in px; the third takes the rest.
  for (const handle of band.querySelectorAll(".resizer-col")) {
    const which = handle.dataset.col; // "1" or "2"
    dragHandle(handle, "x", (e) => {
      const rect = band.getBoundingClientRect();
      if (which === "1") {
        const w = Math.max(200, Math.min(rect.width - 420, e.clientX - rect.left));
        band.style.setProperty("--band-col1", w + "px");
      } else {
        // col2 handle: keep col1 fixed (use its actual rendered width), size col2
        // from the first divider to the pointer.
        const c1w = $("aoi-controls").getBoundingClientRect().width;
        const x = e.clientX - rect.left - c1w - 1;
        const w = Math.max(200, Math.min(rect.width - c1w - 220, x));
        band.style.setProperty("--band-col2", w + "px");
      }
    });
  }

  // Review workspace: its own maps-row height + decision/Sankey column split,
  // mirroring the Harmonize handles above.
  const rev = $("review-workspace");
  const revRowHandle = $("rev-resize-rows");
  if (rev && revRowHandle) {
    dragHandle(revRowHandle, "y", (e) => {
      const rect = rev.getBoundingClientRect();
      const h = Math.max(120, Math.min(rect.height - 160, e.clientY - rect.top));
      rev.style.setProperty("--maps-height", h + "px");
    });
  }
  // Review bottom band: three columns (patches | decision | Sankey), two dividers.
  // Pin the two left column widths in px; the third takes the rest — same scheme as
  // the Harmonize band above.
  const revBand = $("rev-band");
  if (revBand && rev) {
    for (const handle of rev.querySelectorAll(".resizer-col[data-revcol]")) {
      const which = handle.dataset.revcol; // "1" or "2"
      dragHandle(handle, "x", (e) => {
        const rect = revBand.getBoundingClientRect();
        if (which === "1") {
          const w = Math.max(200, Math.min(rect.width - 420, e.clientX - rect.left));
          revBand.style.setProperty("--rev-band-col1", w + "px");
        } else {
          const c1w = $("rev-patches-panel").getBoundingClientRect().width;
          const x = e.clientX - rect.left - c1w - 1;
          const w = Math.max(200, Math.min(rect.width - c1w - 220, x));
          revBand.style.setProperty("--rev-band-col2", w + "px");
        }
      });
    }
  }
}

// Generic pointer-drag wiring shared by every handle.
function dragHandle(el, axis, onMove) {
  if (!el) return;
  el.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    el.setPointerCapture(e.pointerId);
    el.classList.add("dragging");
    document.body.style.cursor = axis === "x" ? "col-resize" : "row-resize";
    const move = (ev) => {
      onMove(ev);
      syncSizes();
    };
    const up = (ev) => {
      el.releasePointerCapture(e.pointerId);
      el.classList.remove("dragging");
      document.body.style.cursor = "";
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      syncSizes();
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  });
}

// ===========================================================================
// Review mode (Stage 6c — docs/PIPELINE.md 6.7). Frontend only: the Harmonize/
// Review mode switch, the table→review handoff, the two large synced inspector
// maps, the left rail (class-pair dropdowns + patch thumbnail index), the right
// rail (candidate edges + multi-select confirm + provenance), and the basemap/
// year switcher. Consumes only the 6b endpoints (/api/review/*) — no new backend.
// "Shared live state" is file-backed: a confirm POST persists to cache/ and the
// reviewed table is recomputed on the next fetch, so both modes read the same
// on-disk table (docs/PIPELINE.md 6.7).
// ===========================================================================

// --- Multi-source satellite basemaps for Review (§6.4). No API key. --------
// Each entry: an XYZ template, optional {years} → the year switcher populates and
// substitutes {year} (ESRI Wayback / Sentinel-2 cloudless). These are the true-
// colour imagery under the queried pixel; the sample-point boxes overlay on top.
const BASEMAPS = {
  google: {
    name: "Google Satellite",
    url: "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attribution: "© Google",
    maxZoom: 21,
  },
  esri: {
    name: "ESRI World Imagery",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "© Esri",
    maxZoom: 19,
  },
  esri_wayback: {
    name: "ESRI Wayback (year-end)",
    url: "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/{waybackid}/{z}/{y}/{x}",
    attribution: "© Esri Wayback",
    maxZoom: 19,
    waybackByYear: {
      2018: 14829, 2019: 20325, 2020: 26993, 2021: 1179,
      2022: 24007, 2023: 47963, 2024: 10, 2025: 20654,
    },
    years: [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
  },
  s2cloudless: {
    name: "Sentinel-2 cloudless",
    url: "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-{year}_3857/default/g/{z}/{y}/{x}.jpg",
    attribution: "Sentinel-2 cloudless © EOX",
    maxZoom: 16,
    years: [2018, 2019, 2020, 2021, 2022, 2023, 2024],
  },
};
const DEFAULT_BASEMAP = "google";

const REV = {
  maps: { ref: null, cmp: null }, // Leaflet inspector maps
  base: { ref: null, cmp: null }, // current satellite basemap layer per side
  outline: { ref: null, cmp: null }, // queried-pixel outline per side
  markers: { ref: null, cmp: null }, // patch location markers per side
  aoi: { ref: null, cmp: null }, // AOI rectangle per side (same box as Harmonize)
  legend: { ref: null, cmp: null }, // [{value,name,color,description}] per side
  pid: { ref: null, cmp: null }, // product id whose legend drives each side
  refValue: null, // reference class focused (drives decision/explore)
  cmpValue: null, // compare class focused (null = "all"/row mode)
  table: null, // reviewed table rows for the current pair
  absent: { ref: [], cmp: [] }, // classes never modelled here (Stage 7), per side
  result: null, // last explorer result (patch locations)
  sankey: null, // ECharts instance on #rev-sankey
  activePatch: -1,
  syncing: false,
  fitted: false, // whether the maps have been fitted to the AOI yet
  initialized: false,
};

function reviewPair() {
  return { reference_id: $("reference").value, compare_id: $("compare").value };
}

// Stand up the two inspector maps once, synchronized, over satellite imagery.
// The legend chips drive the review (focus class + description); the maps show
// true-colour imagery so an expert can eyeball the ground under each sample pixel.
function setupReviewMaps() {
  if (REV.maps.ref) return;
  REV.maps.ref = L.map("rev-map-ref", { worldCopyJump: true }).setView([16, 32], 6);
  REV.maps.cmp = L.map("rev-map-cmp", { worldCopyJump: true }).setView([16, 32], 6);
  REV.outline = { ref: L.layerGroup().addTo(REV.maps.ref), cmp: L.layerGroup().addTo(REV.maps.cmp) };
  REV.markers = { ref: L.layerGroup().addTo(REV.maps.ref), cmp: L.layerGroup().addTo(REV.maps.cmp) };
  REV.aoi = { ref: L.layerGroup().addTo(REV.maps.ref), cmp: L.layerGroup().addTo(REV.maps.cmp) };

  // Lock the two inspector maps together so both show the same ground.
  const sync = (src, dst) =>
    src.on("move", () => {
      if (REV.syncing) return;
      REV.syncing = true;
      dst.setView(src.getCenter(), src.getZoom(), { animate: false });
      REV.syncing = false;
    });
  sync(REV.maps.ref, REV.maps.cmp);
  sync(REV.maps.cmp, REV.maps.ref);

  // Basemap pickers + year switchers.
  fillBasemapSelect("ref");
  fillBasemapSelect("cmp");
  applyBasemap("ref");
  applyBasemap("cmp");
  $("rev-basemap-ref").addEventListener("change", () => onBasemapChange("ref"));
  $("rev-basemap-cmp").addEventListener("change", () => onBasemapChange("cmp"));
  $("rev-year-ref").addEventListener("change", () => applyBasemap("ref"));
  $("rev-year-cmp").addEventListener("change", () => applyBasemap("cmp"));
  REV.initialized = true;
}

function fillBasemapSelect(side) {
  const sel = $(`rev-basemap-${side}`);
  sel.innerHTML = "";
  for (const [key, b] of Object.entries(BASEMAPS)) {
    const o = document.createElement("option");
    o.value = key;
    o.textContent = b.name;
    sel.appendChild(o);
  }
  sel.value = DEFAULT_BASEMAP;
  onBasemapChange(side);
}

// Show/populate the year dropdown (in the title bar) only for basemaps that carry
// snapshots; hide it and its separator otherwise.
function onBasemapChange(side) {
  const b = BASEMAPS[$(`rev-basemap-${side}`).value];
  const ySep = $(`rev-year-sep-${side}`);
  const ySel = $(`rev-year-${side}`);
  if (b.years) {
    ySel.innerHTML = "";
    for (const y of b.years) {
      const o = document.createElement("option");
      o.value = y;
      o.textContent = y;
      ySel.appendChild(o);
    }
    ySel.value = b.years[b.years.length - 1];
    ySel.hidden = false;
    ySep.hidden = false;
  } else {
    ySel.hidden = true;
    ySep.hidden = true;
  }
  applyBasemap(side);
}

// Swap the satellite basemap tile layer for one inspector map.
function applyBasemap(side) {
  const map = REV.maps[side];
  if (!map) return;
  const b = BASEMAPS[$(`rev-basemap-${side}`).value];
  if (REV.base[side]) map.removeLayer(REV.base[side]);
  let url = b.url;
  if (b.years) url = url.replace("{year}", $(`rev-year-${side}`).value);
  if (b.waybackByYear) {
    const wid = b.waybackByYear[$(`rev-year-${side}`).value] || Object.values(b.waybackByYear)[0];
    url = url.replace("{waybackid}", wid);
  }
  // maxZoom 22 with maxNativeZoom at the source's cap lets the map over-zoom
  // (upscaled tiles) far enough that a ~30 m pixel context can fill the view.
  REV.base[side] = L.tileLayer(url, {
    attribution: b.attribution, maxZoom: 22, maxNativeZoom: b.maxZoom,
  }).addTo(map);
  REV.base[side].bringToBack();
}

// --- Mode switch -----------------------------------------------------------

// Pure mode switch: shows/hides the workspaces and stands the Review maps up.
// It must NOT call enterReview — enterReview calls setMode, and the two calling
// each other recursed until the stack overflowed, then every unwound async
// frame still ran its body (thousands of legend/table refetches per switch).
function setMode(mode) {
  document.body.dataset.mode = mode;
  $("mode-harmonize").classList.toggle("active", mode === "harmonize");
  $("mode-review").classList.toggle("active", mode === "review");
  $("mode-harmonize").setAttribute("aria-selected", mode === "harmonize");
  $("mode-review").setAttribute("aria-selected", mode === "review");
  $("review-workspace").hidden = mode !== "review";
  if (mode === "review") {
    setupReviewMaps();
    // Leaflet + ECharts need a resize once their containers become visible.
    setTimeout(() => {
      REV.maps.ref.invalidateSize();
      REV.maps.cmp.invalidateSize();
      if (REV.sankey) REV.sankey.resize();
    }, 0);
  }
}

// Enter Review focused on a reference class (from the table handoff, or the last
// focus when the user just toggles the mode). Loads legends, fits to the AOI, and
// the reviewed table, then renders the decision + Sankey.
async function enterReview(refValue) {
  setMode("review"); // idempotent; sets data-mode and stands maps up
  if (refValue != null) REV.refValue = refValue;
  const pair = reviewPair();
  $("review-context").textContent = `${productName(pair.reference_id)} × ${productName(pair.compare_id)}`;
  $("rev-table-csv").href =
    `/api/review/export?reference_id=${pair.reference_id}&compare_id=${pair.compare_id}`;

  // Legends drive the chips, description strips, and patch/edge swatches. Reuse
  // the Harmonize legends if already loaded for the same products.
  REV.legend.ref = CURRENT_PID.ref === pair.reference_id ? LEGENDS.ref : await fetchLegend(pair.reference_id);
  REV.legend.cmp = CURRENT_PID.cmp === pair.compare_id ? LEGENDS.cmp : await fetchLegend(pair.compare_id);
  REV.pid.ref = pair.reference_id;
  REV.pid.cmp = pair.compare_id;
  $("rev-ref-label").textContent = productName(pair.reference_id);
  $("rev-cmp-label").textContent = productName(pair.compare_id);

  REV.cmpValue = null;
  fillClassSelect("ref");
  fillClassSelect("cmp");
  // Fit to the AOI only on first entry or when the AOI itself changed —
  // toggling Harmonize/Review must keep the user's current pan/zoom.
  const aoiKey = JSON.stringify(readAoiSafe());
  if (!REV.fitted || aoiKey !== REV.fittedAoi) {
    fitReviewToAoi(); // only load imagery for the AOI area, not the whole world
    REV.fittedAoi = aoiKey;
  } else {
    drawReviewAoi(readAoiSafe()); // keep the AOI rectangle in sync anyway
  }

  await loadReviewedTable();
}

// Fit both inspector maps to the current AOI so only AOI-area imagery ever loads.
// A small AOI (e.g. 30x30 km) should fill the view, so pad tightly and let the
// maps zoom in as far as the box allows. Falls back to the Harmonize view when
// no AOI is set.
function fitReviewToAoi() {
  const aoi = readAoiSafe();
  drawReviewAoi(aoi);
  REV.syncing = true;
  if (aoi) {
    const bounds = L.latLngBounds([aoi[1], aoi[0]], [aoi[3], aoi[2]]);
    const opts = { padding: [8, 8], maxZoom: 16 };
    REV.maps.ref.fitBounds(bounds, opts);
    REV.maps.cmp.fitBounds(bounds, opts);
  } else if (MAPS.ref) {
    REV.maps.ref.setView(MAPS.ref.getCenter(), MAPS.ref.getZoom(), { animate: false });
    REV.maps.cmp.setView(MAPS.ref.getCenter(), MAPS.ref.getZoom(), { animate: false });
  }
  REV.syncing = false;
  REV.fitted = true;
}

// Mirror EVERY AOI of the run onto both inspector maps — the primary (red)
// plus each auxiliary (blue; grey-dashed when unused), tooltipped with its
// name — so the expert sees where each piece of evidence comes from: an
// auxiliary-evidenced class's patches fly to ITS box, not the primary's.
// (fill:false keeps only the stroke interactive, so panning/patch clicks
// inside a box are unaffected.)
function drawReviewAoi(aoi) {
  if (!REV.aoi.ref) return;
  REV.aoi.ref.clearLayers();
  REV.aoi.cmp.clearLayers();
  const addBoth = (bbox, style, label) => {
    if (!bbox) return;
    const [w, s, e, n] = bbox;
    for (const side of ["ref", "cmp"]) {
      L.rectangle([[s, w], [n, e]], style)
        .bindTooltip(label, { sticky: true })
        .addTo(REV.aoi[side]);
    }
  };
  addBoth(
    aoi,
    { color: "#c53030", weight: 2, fill: false, dashArray: "4 3" },
    "primary AOI"
  );
  for (const a of (SERVER_AOIS && SERVER_AOIS.auxiliaries) || []) {
    addBoth(
      a.bbox,
      a.disabled
        ? { color: "#7d8896", weight: 2, fill: false, dashArray: "2 5" }
        : { color: "#4b93f7", weight: 2, fill: false, dashArray: "4 3" },
      `auxiliary AOI “${a.name}”` + (a.disabled ? " (unused)" : "")
    );
  }
}

async function fetchLegend(pid) {
  try {
    return (await getJSON(`/api/legend/${pid}`)).classes;
  } catch (_) {
    return null;
  }
}

// Was this class absent from the run (never modelled here)? Stage 7: absent
// classes stay selectable on both sides, but are marked so the expert knows there
// is no evidence behind them.
function isAbsentClass(side, value) {
  const list = (REV.absent && REV.absent[side]) || [];
  return list.some((a) => a.class_value === value);
}

// Fill one side's class dropdown from its legend and wire selection. Choosing a
// class focuses it (drives the decision + evidence) and shows its description
// beneath the picker. The satellite map itself is unchanged.
function fillClassSelect(side) {
  const sel = $(`rev-class-${side}`);
  sel.innerHTML = "";
  // Compare side allows "all" (row mode); reference must pick a class.
  if (side === "cmp") {
    const o = document.createElement("option");
    o.value = "";
    o.textContent = "— all compare classes —";
    sel.appendChild(o);
  }
  for (const c of REV.legend[side] || []) {
    const o = document.createElement("option");
    o.value = c.value;
    // Mark absent classes in the picker (Stage 7): they stay selectable — an
    // expert may declare an edge the AOI cannot evidence — but the expert should
    // see before selecting that this class was never modelled here.
    o.textContent = `${c.name} (${c.value})` + (isAbsentClass(side, c.value) ? " — absent" : "");
    sel.appendChild(o);
  }
  const focus = side === "ref" ? REV.refValue : REV.cmpValue;
  sel.value = focus == null ? "" : String(focus);
  sel.onchange = () => {
    const v = sel.value === "" ? null : Number(sel.value);
    if (side === "ref") REV.refValue = v;
    else REV.cmpValue = v;
    showClassDescription(side, v);
    renderDecision(); // decision follows the focused reference class
  };
  showClassDescription(side, focus);
}

// Set one map's floating top-right class badge (swatch + name).
function setClassBadge(side, value, name) {
  const el = $(`rev-focus-${side}`);
  if (!el) return;
  el.innerHTML = "";
  if (value == null && !name) {
    el.textContent = "no class";
    return;
  }
  el.appendChild(swatchFor(side, value));
  el.appendChild(document.createTextNode(" " + (name ?? "—")));
}

// Write one class's description into the per-side strip beneath the picker, and
// mirror the focused class into the map's top-right badge.
function showClassDescription(side, value) {
  const el = $(side === "ref" ? "rev-desc-ref" : "rev-desc-cmp");
  const c = value == null ? null : (REV.legend[side] || []).find((x) => x.value === value);
  setClassBadge(side, c ? c.value : null, c ? `${c.name} (${c.value})` : null);
  if (!c) {
    el.innerHTML = `<p class="note">Select a class to read its definition.</p>`;
    return;
  }
  el.innerHTML = "";
  const head = document.createElement("div");
  head.className = "rev-desc-head";
  const sw = document.createElement("span");
  sw.className = "swatch";
  sw.style.background = "#" + c.color;
  const name = document.createElement("span");
  name.className = "rev-desc-name";
  name.textContent = `${c.name} (${c.value})`;
  head.append(sw, name);
  const text = document.createElement("div");
  text.className = "rev-desc-text";
  text.textContent = c.description || "No description available for this class.";
  el.append(head, text);
}

// Fetch the reviewed matching table (per-edge provenance) and render the decision
// rail for the focused reference class.
async function loadReviewedTable() {
  const rail = $("rev-edges");
  const pair = reviewPair();
  try {
    const data = await getJSON(
      `/api/review/table?reference_id=${pair.reference_id}&compare_id=${pair.compare_id}`
    );
    REV.table = data.rows;
    // Absent classes per side (Stage 7) drive the "— absent" marks in the class
    // pickers; they stay selectable so an expert can still declare an edge.
    REV.absent = data.absent || { ref: [], cmp: [] };
    fillClassSelect("ref");
    fillClassSelect("cmp");
    // A fresh table load ends the post-retrain comparison window.
    REV.prevProbs = null;
    const report = $("rev-retrain-report");
    if (report) report.hidden = true;
  } catch (e) {
    rail.innerHTML = `<p class="note error">${escapeHtml(e.message)}</p>`;
    $("rev-confirm-row").hidden = true;
    return;
  }
  renderDecision();
}

// Render the candidate edges for the focused reference class: each selectable, with
// its re-balanced probability and provenance (confirmed-frozen vs open). A folded
// "+ more classes" control lets the expert reach ANY compare class, not just the
// algorithm's top candidates (the backend accepts any valid class).
function renderDecision() {
  const rail = $("rev-edges");
  rail.innerHTML = "";
  const rv = REV.refValue;
  const row = (REV.table || []).find((r) => r.reference_value === rv);
  // Always keep the Sankey in step with the current table.
  drawReviewSankey();
  if (rv == null) {
    rail.innerHTML = `<p class="note">Click a reference class chip above to focus it.</p>`;
    $("rev-confirm-row").hidden = true;
    return;
  }
  if (!row) {
    rail.innerHTML = `<p class="note">No reviewed row for this class yet — run a harmonization first.</p>`;
    $("rev-confirm-row").hidden = true;
    return;
  }
  // An absent class was never modelled here, so there is no affinity ranking to
  // guide the expert (Stage 7). It stays selectable and confirmable — the expert
  // may know a correspondence the AOI cannot evidence — but the rail says why the
  // candidates are unranked rather than showing a bare empty list.
  const isAbsent = row.status === "absent";
  if (isAbsent) {
    const why =
      row.absence_reason === "too_rare"
        ? "too few samples in this AOI to model"
        : "no pixels of this class in this AOI";
    $("rev-decision-note").innerHTML =
      `Reference <strong>${escapeHtml(row.reference_name)}</strong> — ` +
      `<span class="badge absent">absent</span> (${escapeHtml(why)}). ` +
      `No affinity ranking exists for this class; add an AOI that covers it, ` +
      `or confirm a correspondence from the legend definitions below.`;
  } else {
    $("rev-decision-note").innerHTML =
      `Reference <strong>${escapeHtml(row.reference_name)}</strong> — ` +
      `<span class="badge ${row.status}">${row.status}</span>. ` +
      `Affinity ranking is guidance only.`;
  }

  const shown = new Set();
  for (const e of row.edges) {
    rail.appendChild(makeEdgeRow(e.compare_value, e.compare_name, e.probability, e.provenance === "expert-confirmed"));
    shown.add(e.compare_value);
  }
  $("rev-confirm-row").hidden = false;

  // A saved "no matching class" decision is shown as a checked row, like any
  // other confirmed choice; otherwise the option lives folded in more-classes.
  const noMatchDecided = !!row.complete && !row.has_confirmed;
  if (noMatchDecided) rail.appendChild(makeNoMatchRow(true));

  // Fix 2: fold every other compare class behind a disclosure so the expert can
  // add and confirm a class the algorithm ranked low. "No matching class" is
  // its last entry.
  const others = (REV.legend.cmp || []).filter((c) => !shown.has(c.value));
  if (others.length || !noMatchDecided) {
    rail.appendChild(makeMoreClasses(others, shown, !noMatchDecided, isAbsent));
  }
}

// The "no matching class" decision row — mutually exclusive with every class
// checkbox: checking it clears them, and checking any class clears it.
function makeNoMatchRow(checked) {
  const div = document.createElement("div");
  div.className = "edge-row nomatch" + (checked ? " confirmed" : "");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.id = "rev-nomatch";
  cb.checked = checked;
  cb.addEventListener("change", () => {
    if (cb.checked) {
      for (const b of $("rev-edges").querySelectorAll('input[type="checkbox"][data-value]')) {
        b.checked = false;
      }
    }
    drawReviewSankey();
  });
  const name = document.createElement("span");
  name.className = "edge-name";
  name.textContent = "∅ No matching class";
  name.title = "This reference class has no corresponding class in the compare map.";
  div.append(cb, name);
  return div;
}

// One selectable candidate-edge row (checkbox + swatch/name + prob + provenance).
function makeEdgeRow(compareValue, compareName, probability, confirmed) {
  const div = document.createElement("div");
  div.className = "edge-row" + (confirmed ? " confirmed" : "");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = confirmed;
  cb.dataset.value = compareValue;
  // Live decision preview: any toggle immediately re-shapes the Sankey.
  // Picking a class also clears the "no matching class" choice.
  cb.addEventListener("change", () => {
    const nm = $("rev-nomatch");
    if (cb.checked && nm) nm.checked = false;
    drawReviewSankey();
  });
  const sw = swatchFor("cmp", compareValue);
  const name = document.createElement("span");
  name.className = "edge-name";
  name.appendChild(sw);
  name.appendChild(document.createTextNode(" " + compareName));
  const c = (REV.legend.cmp || []).find((x) => x.value === compareValue);
  if (c && c.description) name.title = c.description;
  const prob = document.createElement("span");
  prob.className = "edge-prob";
  prob.textContent = probability == null ? "—" : fmt(probability, 2);
  const prov = document.createElement("span");
  prov.className = "prov " + (confirmed ? "confirmed" : "open");
  prov.textContent = confirmed ? "confirmed" : "open";
  div.append(cb, name, prob);
  // Post-retrain delta on open candidates: how this probability moved.
  if (!confirmed && probability != null && REV.prevProbs) {
    const prev = REV.prevProbs[REV.refValue];
    const old = prev ? prev[String(compareValue)] : undefined;
    if (old != null) {
      const dp = probability - Number(old);
      if (Math.abs(dp) >= 0.005) {
        const d = document.createElement("span");
        d.className = "delta " + (dp > 0 ? "up" : "down");
        d.textContent = `${dp > 0 ? "↑" : "↓"}${dp > 0 ? "+" : ""}${dp.toFixed(2)}`;
        d.title = `Changed by retrain: was ${fmt(Number(old), 2)}`;
        div.appendChild(d);
      }
    }
  }
  div.appendChild(prov);
  return div;
}

// The algorithm's re-balanced probability for ANY compare class of the focused
// reference row: kept edges carry it directly; every other class comes from the
// row's full probability map (all_probabilities). Null when truly unknown (e.g.
// a class absent from the affinity matrix).
function probFor(row, compareValue) {
  if (!row) return null;
  const known = (row.edges || []).find((e) => e.compare_value === compareValue);
  if (known) return known.probability;
  const p = row.all_probabilities && row.all_probabilities[String(compareValue)];
  return p === undefined ? null : p;
}

// The folded "+ more classes" disclosure (Fix 2): a <details> listing every other
// compare class WITH its affinity value; picking one appends a checked,
// confirmable edge row for it.
function makeMoreClasses(others, shown, includeNoMatch, openByDefault = false) {
  const det = document.createElement("details");
  det.className = "more-classes";
  // An absent class has no ranked edges at all, so every candidate lives in here:
  // folding it shut would leave the rail looking empty and unactionable.
  det.open = openByDefault;
  const sum = document.createElement("summary");
  sum.textContent = `+ more classes (${others.length + (includeNoMatch ? 1 : 0)})`;
  det.appendChild(sum);
  const row = (REV.table || []).find((r) => r.reference_value === REV.refValue);
  for (const c of others) {
    const p = probFor(row, c.value);
    const opt = document.createElement("div");
    opt.className = "more-option";
    const sw = swatchFor("cmp", c.value);
    const label = document.createElement("span");
    label.textContent = ` ${c.name} (${c.value})`;
    const prob = document.createElement("span");
    prob.className = "edge-prob";
    prob.textContent = p == null ? "—" : fmt(p, 2);
    if (c.description) opt.title = c.description;
    opt.append(sw, label, prob);
    opt.addEventListener("click", () => {
      if (shown.has(c.value)) return;
      shown.add(c.value);
      const newRow = makeEdgeRow(c.value, c.name, p, false);
      newRow.querySelector('input[type="checkbox"]').checked = true;
      $("rev-edges").insertBefore(newRow, det);
      opt.remove();
      if (!det.querySelector(".more-option")) det.remove();
      drawReviewSankey(); // programmatic check fires no change event

    });
    det.appendChild(opt);
  }
  // Last entry: "no matching class". Picking it promotes a checked no-match
  // row into the rail (clearing every class selection), like any other option.
  if (includeNoMatch) {
    const nm = document.createElement("div");
    nm.className = "more-option nomatch";
    nm.textContent = "∅ No matching class";
    nm.title = "This reference class has no corresponding class in the compare map.";
    nm.addEventListener("click", () => {
      for (const b of $("rev-edges").querySelectorAll('input[type="checkbox"][data-value]')) {
        b.checked = false;
      }
      $("rev-edges").insertBefore(makeNoMatchRow(true), det);
      nm.remove();
      if (!det.querySelector(".more-option")) det.remove();
      drawReviewSankey();
    });
    det.appendChild(nm);
  }
  return det;
}

// A small colour swatch (legend DATA) for a class value on one side.
function swatchFor(side, value) {
  const sw = document.createElement("span");
  sw.className = "swatch";
  const legend = REV.legend[side];
  const c = legend && legend.find((x) => x.value === value);
  if (c) sw.style.background = "#" + c.color;
  return sw;
}

// --- Explorer: run a three-mode evidence query and index the patches -------

async function runExplore() {
  const status = $("rev-explore-status");
  const pair = reviewPair();
  const refSel = REV.refValue;
  const cmpSel = REV.cmpValue;
  // Mode from which side(s) are focused: both → cell, reference-only → row,
  // compare-only → column (docs/PIPELINE.md 6.2).
  let mode, body = { ...pair };
  if (cmpSel != null && refSel != null) {
    mode = "both";
    body.reference_value = Number(refSel);
    body.compare_value = Number(cmpSel);
  } else if (refSel != null) {
    mode = "reference";
    body.reference_value = Number(refSel);
  } else if (cmpSel != null) {
    mode = "compare";
    body.compare_value = Number(cmpSel);
  } else {
    status.className = "progress error";
    status.textContent = "Focus a reference and/or compare class chip first.";
    return;
  }
  body.mode = mode;
  body.aoi = readAoi(); // share the Harmonize AOI
  const nPts = Number($("rev-n").value);
  if (nPts >= 1) body.n = Math.round(nPts); // user-chosen number of points
  const ov = Number($("rev-oversample").value);
  if (ov >= 1) body.oversample = ov; // live-path candidate oversample factor

  status.className = "progress";
  status.textContent = "Finding evidence…";
  $("rev-explore").disabled = true;
  try {
    REV.result = await getJSON("/api/review/explore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderPatches();
    const src = REV.result.source === "cache" ? "training sample points" : "live query";
    status.textContent = `${REV.result.locations.length} location(s) — ${src}.`;
  } catch (e) {
    status.className = "progress error";
    status.textContent = "Explore failed: " + e.message;
  } finally {
    $("rev-explore").disabled = false;
  }
}

// The patch thumbnail index (left rail): each location is a card that flies both
// synced maps to it and outlines the queried pixel.
function renderPatches() {
  const wrap = $("rev-patches");
  wrap.innerHTML = "";
  REV.activePatch = -1;
  // A fresh evidence set re-arms the one-time fit: the FIRST point zooms to
  // the pixel window, later points pan at whatever zoom the user has set.
  REV.patchZoomed = false;
  const locs = (REV.result && REV.result.locations) || [];
  if (!locs.length) {
    wrap.innerHTML = `<p class="note">No qualifying locations found for this query.</p>`;
    return;
  }
  locs.forEach((loc, i) => {
    const card = document.createElement("div");
    card.className = "patch-card";
    const coord = document.createElement("div");
    coord.className = "coord";
    coord.textContent = `${loc.lat.toFixed(4)}, ${loc.lon.toFixed(4)}`;
    const labels = document.createElement("div");
    labels.className = "labels";
    labels.append(
      patchLabel("ref", loc.reference_label, loc.reference_label_name),
      patchLabel("cmp", loc.compare_label, loc.compare_label_name)
    );
    card.append(coord, labels);
    card.addEventListener("click", () => flyToPatch(i));
    wrap.appendChild(card);
  });
  drawAllOutlines();
}

// The lat/lon bounds of one location's single-pixel box on one side.
function pixelBoundsFor(loc, pixelM) {
  const half = pixelM / 2;
  const dLat = half / 111320;
  const dLon = half / (111320 * Math.cos((loc.lat * Math.PI) / 180));
  return [[loc.lat - dLat, loc.lon - dLon], [loc.lat + dLat, loc.lon + dLon]];
}

const OUTLINE_DIM = { color: "#f6c744", weight: 1, opacity: 0.6, fill: false };
const OUTLINE_ACTIVE = { color: "#f6c744", weight: 3, opacity: 1, fill: false };

// Draw EVERY evidence point's pixel box on both maps (dim), so the whole sample
// stays visible; flyToPatch only re-styles the active one, never removes boxes.
// Clicking a box on the map selects that point, same as clicking its card.
function drawAllOutlines() {
  const locs = (REV.result && REV.result.locations) || [];
  const pixelM = {
    ref: (REV.result && REV.result.reference_pixel_m) || 10,
    cmp: (REV.result && REV.result.compare_pixel_m) || 10,
  };
  REV.outlineRects = { ref: [], cmp: [] };
  for (const side of ["ref", "cmp"]) {
    REV.outline[side].clearLayers();
    locs.forEach((loc, i) => {
      const rect = L.rectangle(pixelBoundsFor(loc, pixelM[side]), OUTLINE_DIM);
      rect.on("click", () => flyToPatch(i));
      rect.addTo(REV.outline[side]);
      REV.outlineRects[side].push(rect);
    });
  }
}

function patchLabel(side, value, name) {
  const span = document.createElement("span");
  span.className = "lab";
  span.appendChild(swatchFor(side, value));
  span.appendChild(document.createTextNode(" " + (name ?? "—")));
  const c = (REV.legend[side] || []).find((x) => x.value === value);
  if (c && c.description) span.title = c.description;
  return span;
}

// Selecting a patch flies BOTH synced maps to it and outlines the queried pixel's
// label footprint on each side (§6.3).
function flyToPatch(i) {
  const loc = REV.result.locations[i];
  if (!loc) return;
  const prevPatch = REV.activePatch; // where the eye currently is (see below)
  REV.activePatch = i;
  $("rev-patches").querySelectorAll(".patch-card").forEach((c, j) => {
    c.classList.toggle("active", j === i);
    if (j === i) c.scrollIntoView({ block: "nearest" });
  });

  const pixelM = {
    ref: REV.result.reference_pixel_m || 10,
    cmp: REV.result.compare_pixel_m || 10,
  };

  REV.syncing = true;
  if (!REV.patchZoomed) {
    // FIRST point of this evidence set only: zoom so the pixel box spans ~1/3
    // of the window height (a bounds three pixel-widths tall centred on the
    // point). After that the user's own zoom is respected — stepping to the
    // next point just pans there at the current zoom.
    const half = Math.max(pixelM.ref, pixelM.cmp) * 1.5;
    const cLat = half / 111320;
    const cLon = half / (111320 * Math.cos((loc.lat * Math.PI) / 180));
    const viewBounds = [
      [loc.lat - cLat, loc.lon - cLon],
      [loc.lat + cLat, loc.lon + cLon],
    ];
    REV.maps.ref.fitBounds(viewBounds, { animate: true });
    REV.maps.cmp.fitBounds(viewBounds, { animate: true });
    REV.patchZoomed = true;
  } else {
    // Keep the point where the eye already is: place the NEW point at the
    // same screen position the PREVIOUS point occupies (the user may have
    // panned it away from center on purpose). Falls back to centering when
    // there is no previous point or it sits off-screen.
    const map = REV.maps.ref;
    const z = map.getZoom();
    let center = L.latLng(loc.lat, loc.lon);
    const prev =
      prevPatch >= 0 && REV.result.locations[prevPatch] !== undefined
        ? REV.result.locations[prevPatch]
        : null;
    if (prev) {
      const p = map.latLngToContainerPoint([prev.lat, prev.lon]);
      const size = map.getSize();
      if (p.x >= 0 && p.y >= 0 && p.x <= size.x && p.y <= size.y) {
        const q = map.latLngToContainerPoint([loc.lat, loc.lon]);
        const c = map.latLngToContainerPoint(map.getCenter());
        center = map.containerPointToLatLng(
          L.point(c.x + (q.x - p.x), c.y + (q.y - p.y))
        );
      }
    }
    REV.maps.ref.setView(center, z, { animate: true });
    REV.maps.cmp.setView(center, z, { animate: true });
  }
  REV.syncing = false;

  // Highlight the active point's pixel box; every other point's box stays on the
  // map (dim) so the whole sample remains visible.
  if (!REV.outlineRects || !REV.outlineRects.ref.length) drawAllOutlines();
  for (const side of ["ref", "cmp"]) {
    REV.outlineRects[side].forEach((r, j) =>
      r.setStyle(j === i ? OUTLINE_ACTIVE : OUTLINE_DIM)
    );
  }

  // Each map's top-right badge shows what THAT map says at this point.
  setClassBadge("ref", loc.reference_label, loc.reference_label_name);
  setClassBadge("cmp", loc.compare_label, loc.compare_label_name);
}

// --- Reviewed-table Sankey (bottom-right) ----------------------------------
// Whole reviewed table: reference classes (left) → compare classes (right).
// A row REFLECTS THE EXPERT'S DECISION: once any edge is decided (checked in the
// focused row, or confirmed on disk), only the decided edges are drawn, with the
// row's whole mass split among them — "Tree cover → only Trees" shows exactly
// that. Undecided rows show the algorithm's re-balanced proposals as before.
// Redrawn on every table change AND on every checkbox toggle (live preview).

// The links to draw for one row: [{value, name, probability}], normalized to
// sum 1 when the row is decided.
function sankeyEdgesFor(row) {
  // The pseudo-edge that keeps a "no matching class" reference visible in the
  // Sankey: a full-width link into a shared grey "no match" node.
  const noMatchEdge = [{ compare_value: null, compare_name: "no matching class", probability: 1 }];

  // Focused row: the checkboxes are the live decision preview.
  let decided = [];
  if (row.reference_value === REV.refValue) {
    const nm = $("rev-nomatch");
    if (nm && nm.checked) return noMatchEdge; // live preview
    const boxes = [...$("rev-edges").querySelectorAll('input[type="checkbox"][data-value]')];
    decided = boxes.filter((b) => b.checked).map((b) => Number(b.dataset.value));
  }
  // Other rows: whatever is confirmed on disk.
  if (!decided.length) {
    decided = (row.edges || [])
      .filter((e) => e.provenance === "expert-confirmed")
      .map((e) => e.compare_value);
  }
  if (!decided.length) {
    // A complete row with no confirmed edges = saved "no matching class".
    return row.complete ? noMatchEdge : row.edges || [];
  }

  const chosen = decided.map((cv) => {
    const known = (row.edges || []).find((e) => e.compare_value === cv);
    const legend = (REV.legend.cmp || []).find((x) => x.value === cv);
    return {
      compare_value: cv,
      compare_name: known ? known.compare_name : legend ? legend.name : String(cv),
      probability: probFor(row, cv) || 0,
    };
  });
  // The decision claims the row: split its whole mass among the decided edges,
  // proportionally to their probabilities (evenly if all are zero).
  const total = chosen.reduce((s, e) => s + e.probability, 0);
  for (const e of chosen) {
    e.probability = total > 0 ? e.probability / total : 1 / chosen.length;
  }
  return chosen;
}

function drawReviewSankey() {
  const el = $("rev-sankey");
  if (!el) return;
  if (!REV.sankey) {
    REV.sankey = echarts.init(el, null, { renderer: "canvas" });
    // Clicking a link focuses both of its classes; clicking a class bar
    // focuses that class — same effect as picking it in the dropdowns, so the
    // decision panel and the class descriptions follow the Sankey.
    REV.sankey.on("click", (p) => {
      const focusByName = (side, label) => {
        const c = (REV.legend[side] || []).find((x) => x.name === label);
        const sel = $(`rev-class-${side}`);
        if (!c || !sel || !sel.onchange) return;
        sel.value = String(c.value);
        sel.onchange();
      };
      if (p.dataType === "edge") {
        focusByName("cmp", p.data.target.replace(/^cmp:/, ""));
        focusByName("ref", p.data.source.replace(/^ref:/, ""));
      } else if (typeof p.name === "string" && p.name.startsWith("ref:")) {
        focusByName("ref", p.name.slice(4));
      } else if (typeof p.name === "string" && p.name.startsWith("cmp:")) {
        focusByName("cmp", p.name.slice(4));
      }
    });
  }

  const nodes = [];
  const seen = new Set();
  const addNode = (id, label, color, side) => {
    if (seen.has(id)) return;
    seen.add(id);
    const item = {
      name: id,
      label: { formatter: label, position: side === "cmp" ? "left" : "right" },
    };
    if (color) item.itemStyle = { color };
    nodes.push(item);
  };
  const revColor = (side, value) => {
    const c = (REV.legend[side] || []).find((x) => x.value === value);
    return c ? "#" + c.color : null;
  };

  const links = [];
  for (const row of REV.table || []) {
    const edges = sankeyEdgesFor(row);
    if (!edges.length) continue;
    const refId = "ref:" + row.reference_name;
    addNode(refId, row.reference_name, revColor("ref", row.reference_value), "ref");
    for (const e of edges) {
      if (!e.probability || e.probability <= 0.001) continue;
      // "No matching class": keep the reference bar visible on the left by
      // linking it to a shared INVISIBLE sink — no drawn link, no right node.
      if (e.compare_value == null) {
        if (!seen.has("cmp:∅")) {
          seen.add("cmp:∅");
          nodes.push({
            name: "cmp:∅",
            itemStyle: { opacity: 0 },
            label: { show: false },
            emphasis: { disabled: true },
          });
        }
        links.push({
          source: refId,
          target: "cmp:∅",
          value: e.probability,
          lineStyle: { opacity: 0 },
          emphasis: { disabled: true },
        });
        continue;
      }
      const cmpId = "cmp:" + e.compare_name;
      addNode(cmpId, e.compare_name, revColor("cmp", e.compare_value), "cmp");
      links.push({ source: refId, target: cmpId, value: e.probability });
    }
  }

  REV.sankey.setOption({
    backgroundColor: "transparent",
    tooltip: {
      trigger: "item",
      formatter: (p) => {
        if (p.dataType === "edge") {
          if (p.data.target === "cmp:∅") {
            return `${p.data.source.replace(/^ref:/, "")} — no matching class`;
          }
          return `${p.data.source.replace(/^ref:/, "")} → ${p.data.target.replace(/^cmp:/, "")}<br/>p = ${p.data.value.toFixed(3)}`;
        }
        return p.name === "cmp:∅" ? "" : p.name.replace(/^(ref|cmp):/, "");
      },
    },
    series: [
      {
        type: "sankey",
        left: 12, right: 20, top: 10, bottom: 10,
        nodeGap: 8, nodeWidth: 14,
        emphasis: { focus: "adjacency" },
        data: nodes,
        links: links,
        label: { color: "#e6edf3", fontSize: 11 },
        lineStyle: { color: "gradient", opacity: 0.45, curveness: 0.5 },
      },
    ],
  }, true);
  REV.sankey.resize();
  // Keep the table view (if shown) in step with every table/decision change.
  if (!$("rev-table").hidden) renderRevTable();
}

// --- Revised-table view (shares the Sankey panel; toggled by a button) -----

// Render the legend-matching result as a simple 4-column crosswalk: code and
// name for the reference map, code and name for the compare map — one row per
// mapping edge.
function renderRevTable() {
  const wrap = $("rev-table");
  wrap.innerHTML = "";
  if (!REV.table || !REV.table.length) {
    wrap.innerHTML = `<p class="note">Run a harmonization to build the matching table.</p>`;
    return;
  }
  const pair = reviewPair();
  // Trim the long registry qualifiers, e.g. "ESA WorldCover v200 (global
  // static 2021) [test-swap reference]" → "ESA WorldCover v200".
  const shortName = (pid) => productName(pid).replace(/\s*[\(\[].*$/, "").trim();
  const refName = shortName(pair.reference_id);
  const cmpName = shortName(pair.compare_id);
  const t = document.createElement("table");
  t.className = "matching";
  t.innerHTML =
    `<thead><tr><th>${escapeHtml(refName)} code</th><th>${escapeHtml(refName)} class</th>` +
    `<th>${escapeHtml(cmpName)} code</th><th>${escapeHtml(cmpName)} class</th></tr></thead>`;
  const tb = document.createElement("tbody");
  for (const row of REV.table) {
    // Same rule as the Sankey: once the expert has confirmed edges for a
    // class, the decision IS the mapping — open proposals are not shown.
    const all = row.edges || [];
    const confirmed = all.filter((e) => e.provenance === "expert-confirmed");
    // A complete row with no confirmed edges = "no matching class": show "—".
    const edges = confirmed.length ? confirmed : row.complete ? [] : all;
    // Clicking a row focuses BOTH of its classes (like clicking a Sankey
    // link): the reference class drives the decision panel and both class
    // descriptions follow, via the dropdowns' own change handlers.
    const focus = (cv) => {
      const focusSide = (side, v) => {
        const sel = $(`rev-class-${side}`);
        if (v == null || !sel || !sel.onchange) return;
        sel.value = String(v);
        sel.onchange();
      };
      focusSide("cmp", cv);
      focusSide("ref", row.reference_value);
    };
    const cells = edges.length
      ? edges.map((e) => [e.compare_value, e.compare_name])
      : [[null, row.complete ? "∅ no matching class" : "—"]];
    cells.forEach(([cv, cn], i) => {
      const tr = document.createElement("tr");
      tr.className = "status-" + row.status;
      tr.innerHTML =
        (i === 0
          ? `<td>${row.reference_value}</td><td>${escapeHtml(row.reference_name)}</td>`
          : "<td></td><td></td>") +
        `<td>${cv == null ? "—" : cv}</td><td>${escapeHtml(String(cn))}</td>`;
      tr.classList.add("reviewable");
      tr.title = "Focus this row's classes in the decision panel";
      tr.addEventListener("click", () => focus(cv));
      tb.appendChild(tr);
    });
  }
  t.appendChild(tb);
  wrap.appendChild(t);
}

// Toggle the panel between the Sankey and the revised table.
function toggleRevView() {
  const showTable = $("rev-table").hidden; // about to show the table?
  $("rev-table").hidden = !showTable;
  $("rev-sankey").hidden = showTable;
  $("rev-view-toggle").textContent = showTable ? "Sankey" : "Table";
  if (showTable) renderRevTable();
  else if (REV.sankey) REV.sankey.resize();
}

// --- Confirm / unconfirm edges --------------------------------------------

async function confirmEdges() {
  const status = $("rev-confirm-status");
  const pair = reviewPair();
  const rv = REV.refValue;
  const noMatch = !!($("rev-nomatch") && $("rev-nomatch").checked);
  const boxes = [...$("rev-edges").querySelectorAll('input[type="checkbox"][data-value]')];
  const selected = noMatch
    ? [] // "no matching class": clear everything, save the row as complete
    : boxes.filter((b) => b.checked).map((b) => Number(b.dataset.value));

  // Reconcile with what is already confirmed on disk: newly-checked → confirm,
  // newly-unchecked (previously confirmed) → unconfirm.
  const row = (REV.table || []).find((r) => r.reference_value === rv);
  const wasConfirmed = new Set(
    (row ? row.edges : []).filter((e) => e.provenance === "expert-confirmed").map((e) => e.compare_value)
  );
  const toUnconfirm = [...wasConfirmed].filter((v) => !selected.includes(v));

  status.className = "progress";
  status.textContent = "Saving…";
  $("rev-confirm").disabled = true;
  try {
    // ONE request carries the whole decision (confirms + unconfirms); the backend
    // saves and returns the recomputed table. No retraining happens here — that is
    // the explicit "Retrain model" button.
    const data = await getJSON("/api/review/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...pair,
        reference_value: rv,
        compare_values: selected,
        unconfirm_values: toUnconfirm,
        // Confirm is a plain save: confirmed edges freeze at the algorithm's
        // probability (no renormalising to 1) and stay editable. `complete`
        // is sent ONLY for the explicit "no matching class" decision.
        complete: noMatch,
        refit: false,
      }),
    });
    REV.table = data.rows; // freshly recomputed reviewed table
    renderDecision(); // redraws the edge list AND the Sankey immediately
    status.textContent = "Saved — edges frozen. Retrain when you're done adjusting classes.";
  } catch (e) {
    status.className = "progress error";
    status.textContent = "Confirm failed: " + e.message;
  } finally {
    $("rev-confirm").disabled = false;
  }
}

// Explicit retrain (Stage 6.6): warm-start-refit every confirmed class's GMM and
// re-propose the open edges. Deliberately separate from confirm, which only saves.
async function retrainModel() {
  const status = $("rev-confirm-status");
  const pair = reviewPair();
  status.className = "progress";
  status.textContent = "Retraining…";
  $("rev-retrain").disabled = true;
  try {
    const before = REV.table || [];
    const data = await getJSON("/api/review/refit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...pair, reference_value: null }),
    });
    // Keep the pre-retrain probabilities so the decision rail can show ↑/↓
    // deltas next to each open candidate until the next table reload.
    REV.prevProbs = {};
    for (const r of before) REV.prevProbs[r.reference_value] = r.all_probabilities || {};
    REV.table = data.rows;
    renderDecision();
    const done = (data.refits || []).filter((r) => r.refit).length;
    status.textContent = done
      ? `Retrained ${done} class(es) — open edges re-proposed.`
      : "Nothing to retrain yet — confirm some edges first.";
    renderRetrainReport(before, data.rows, done);
  } catch (e) {
    status.className = "progress error";
    status.textContent = "Retrain failed: " + e.message;
  } finally {
    $("rev-retrain").disabled = false;
  }
}

// The "what changed" summary under the buttons: compare the pre- and post-
// retrain tables and report, per class, a changed top candidate or a
// meaningful probability shift on the open (unconfirmed) proposals.
function renderRetrainReport(before, after, done) {
  const box = $("rev-retrain-report");
  if (!box) return;
  if (!done) {
    box.hidden = true;
    return;
  }
  const oldByRv = new Map(before.map((r) => [r.reference_value, r]));
  const open = (r) => (r.edges || []).filter((e) => e.provenance !== "expert-confirmed");
  const lines = [];
  let unchanged = 0;
  for (const row of after) {
    const old = oldByRv.get(row.reference_value);
    if (!old) continue;
    const nowTop = open(row)[0];
    const wasTop = open(old)[0];
    if (!nowTop || !wasTop) continue; // fully decided rows: nothing open to move
    if (nowTop.compare_value !== wasTop.compare_value) {
      lines.push(
        `<li><strong>${escapeHtml(row.reference_name)}</strong>: top candidate ` +
        `${escapeHtml(wasTop.compare_name)} ${fmt(wasTop.probability, 2)} → ` +
        `<strong>${escapeHtml(nowTop.compare_name)} ${fmt(nowTop.probability, 2)}</strong></li>`
      );
    } else if (Math.abs(nowTop.probability - wasTop.probability) >= 0.02) {
      lines.push(
        `<li><strong>${escapeHtml(row.reference_name)}</strong>: ` +
        `${escapeHtml(nowTop.compare_name)} ${fmt(wasTop.probability, 2)} → ${fmt(nowTop.probability, 2)}</li>`
      );
    } else {
      unchanged += 1;
    }
  }
  box.hidden = false;
  box.innerHTML =
    `<strong>Retrain result</strong> — ` +
    (lines.length
      ? `${lines.length} class(es) with changed proposals, ${unchanged} unchanged:` +
        `<ul>${lines.join("")}</ul>`
      : `no meaningful proposal changes (${unchanged} open class(es) checked).`);
}

// --- Review wiring ---------------------------------------------------------

$("mode-harmonize").addEventListener("click", () => setMode("harmonize"));
// The Review tab enters via enterReview (which itself switches the mode), so
// the workspace is loaded exactly once per entry.
$("mode-review").addEventListener("click", () => enterReview(null));
$("rev-explore").addEventListener("click", runExplore);
$("rev-confirm").addEventListener("click", confirmEdges);
$("rev-retrain").addEventListener("click", retrainModel);
$("rev-view-toggle").addEventListener("click", toggleRevView);

// ← / → step through the evidence points in Review mode (skipped while typing in
// an input/select so the arrow keys keep their normal meaning there).
document.addEventListener("keydown", (e) => {
  if (document.body.dataset.mode !== "review") return;
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  const tag = e.target && e.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  const locs = (REV.result && REV.result.locations) || [];
  if (!locs.length) return;
  e.preventDefault();
  const step = e.key === "ArrowRight" ? 1 : -1;
  const next =
    REV.activePatch < 0
      ? (step > 0 ? 0 : locs.length - 1)
      : (REV.activePatch + step + locs.length) % locs.length;
  flyToPatch(next);
});

// --- Wire up ---------------------------------------------------------------

// Per-AOI controls (bbox inputs, overlap check, upload, Run/Sample) live on the
// cards and are wired in aoiCardEl; only the card-list-level controls are here.
$("run").addEventListener("click", runAll);
// "+ add AOI" appends a draft auxiliary card, same as the absence dialog's CTA.
$("aoi-add").addEventListener("click", addAoiCard);

// ⓘ coverage dialog: which AOI evidences each class + what is still absent.
$("aoi-info").addEventListener("click", showCoverageModal);
$("coverage-continue").addEventListener("click", closeCoverageModal);
$("coverage-add-aoi").addEventListener("click", addAoiCard);
$("coverage-modal").addEventListener("click", (e) => {
  if (e.target.id === "coverage-modal") closeCoverageModal(); // backdrop click
});

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("coverage-modal").hidden) closeCoverageModal();
});
$("csv-pick").addEventListener("change", renderCsvView);

// Switching a product reloads that side's tiles/legend and redraws footprints.
$("reference").addEventListener("change", async () => {
  await refreshMapSide("ref");
  await refreshFootprints();
});
$("compare").addEventListener("change", async () => {
  await refreshMapSide("cmp");
  await refreshFootprints();
});

// Rescan data/ for datasets dropped in while the server was already running.
$("refresh-datasets").addEventListener("click", refreshDatasets);

// Offer to delete the derived files of datasets whose data/ folder is gone.
//
// Confirmed per dataset, naming the exact size, because the amount is not
// trivial (a converted tile set runs to several GB) and the decision is the
// user's: they may have moved the folder aside deliberately and intend to put
// it back, in which case the COGs are worth keeping.
async function offerCleanup(missing) {
  const status = $("map-status");
  for (const d of missing) {
    let info;
    try {
      info = await getJSON(`/api/datasets/${d.product_id}/artifacts`);
    } catch (_) {
      continue;
    }
    if (!info.artifacts.length) continue;
    const ok = window.confirm(
      `“${d.folder}” was removed from data/, but ${info.size} of derived files ` +
        `are still on disk:\n\n` +
        info.artifacts.map((a) => "  • " + a).join("\n") +
        `\n\nDelete them?\n\n` +
        `(Your data/ folder is never touched. Choose Cancel to keep them — ` +
        `restore the folder and press ↻ datasets to use the dataset again.)`
    );
    if (!ok) {
      status.className = "note";
      status.textContent =
        `Kept ${info.size} of derived files for “${d.folder}”. ` +
        `Restore the folder under data/ and press ↻ datasets to use it again.`;
      continue;
    }
    try {
      const res = await getJSON(`/api/datasets/${d.product_id}`, {
        method: "DELETE",
      });
      status.className = "note";
      status.textContent = `Removed “${d.folder}” — ${res.freed} freed.`;
    } catch (e) {
      status.className = "note error";
      status.textContent = `Could not remove “${d.folder}”: ${e.message}`;
    }
  }
  // Reflect the result in the pickers without a reload.
  const data = await getJSON("/api/products");
  const labels = data.products.filter((p) => p.kind === "label");
  fillSelect($("reference"), labels);
  fillSelect($("compare"), labels);
}

// Explain the drop-in naming convention, in the overlap-info panel (the app's
// general "here is what just happened" area). Shown when a dataset is sitting
// at needs-legend, which is exactly when the user needs to know the rule.
async function showDropInRules(pending) {
  const info = $("overlap-info");
  let rules;
  try {
    rules = (await getJSON("/api/datasets")).rules;
  } catch (_) {
    return;
  }
  if (!rules) return;

  const lines = [
    `${pending.map((d) => d.folder).join(", ")} — no legend found.`,
    "",
    "Each dataset is one folder: its rasters, plus its legend beside them.",
    "",
    rules.layout,
    "",
    ...rules.steps.map((s, i) => `  ${i + 1}. ${s}`),
    "",
    `Legend CSV columns: ${rules.legend_columns.join(", ")}`,
    "",
    rules.note,
  ];
  info.className = "info warn";
  info.style.whiteSpace = "pre-wrap";
  info.textContent = lines.join("\n");
}

// (Typing a bounding box is handled per card in aoiCardEl: it updates the drawn
// rectangle + footprints, and — on the primary — re-fits the Review maps.)

// Keep the Sankey sized to its panel; also nudge Leaflet to recompute after resize.
window.addEventListener("resize", syncSizes);
setupResizers();

init()
  // The AOI manager reads server truth (cache/aois.json + GMM caches), so it
  // renders before any run in this session — a prior session's AOIs show too.
  .then(() => refreshAoiList())
  .catch((e) => {
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<p class="info error">Failed to load: ${escapeHtml(e.message)}</p>`
  );
});
