// City map viewer for the files written by "python main.py --export": results.json and maps/
"use strict";

const MAX_SCALE = 32;
const $ = (selector) => document.querySelector(selector);
const panels = $("#panels");
const frames = [...panels.querySelectorAll(".frame")];
const images = Object.fromEntries(frames.map((frame) => [frame.dataset.key, frame.querySelector("img")]));
const tooltip = $("#map-tooltip");

// Scale and offset shared by all panels, the offsets are fractions of the frame size
const view = { s: 1, x: 0, y: 0 };
// Class id of every pixel of the label and prediction maps, by image URL
const classIds = {};
const colorIds = new Map();
let results = null;
let city = null;
let drag = null;

fetch("results.json", { cache: "no-cache" })
  .then((response) => {
    if (!response.ok) throw new Error(`results.json: ${response.status}`);
    return response.json();
  })
  .then(init, () => { $("#no-results").hidden = false; });

function init(data) {
  results = data;
  for (const c of data.classes) colorIds.set(parseInt(c.color.slice(1), 16), c.id);
  for (const c of data.cities) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = c.name;
    button.setAttribute("role", "tab");
    button.addEventListener("click", () => selectCity(c));
    $("#city-tabs").append(button);
  }
  // LCZ 1 to G, then no data
  for (const c of [...data.classes.slice(1), data.classes[0]]) {
    const li = document.createElement("li");
    li.append(swatch(c.id));
    $("#legend").append(li);
  }
  showModel(data.model);
  $("#viewer").hidden = false;
  // Smallest city first, it loads fastest
  selectCity([...data.cities].sort((a, b) => a.patches - b.patches)[0]);

  const hero = $("#hero-figure");
  hero.querySelector("img").addEventListener("load", () => { hero.hidden = false; });
  hero.querySelector("img").src = "maps/overview.png";
}

function className(id) {
  const c = results.classes[id];
  return id === 0 ? c.name : `LCZ ${c.code} ${c.name}`;
}

// Class name with a color chip, the chip is drawn by .swatch::before
function swatch(id) {
  const span = document.createElement("span");
  span.className = "swatch";
  span.style.setProperty("--c", results.classes[id].color);
  span.textContent = className(id);
  return span;
}

function listing(names) {
  return names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names.at(-1)}` : names[0];
}

function showModel(m) {
  const weights = m.pretrained ? "ImageNet weights" : "trained from scratch";
  const epochs = `at most ${m.epochs} epoch${m.epochs === 1 ? "" : "s"}`;
  const seed = m.seed === null ? "" : `, seed ${m.seed}`;
  // Exports from before these fields existed don't have them
  const cities = m.train_cities ? ` Trained on the training patches of ${listing(m.train_cities)}, ` +
    `tested on the test patches of ${listing(m.test_cities)}.` : "";
  $("#model-info").textContent = `Model: ${m.description} (${weights}), ${epochs}, batch size ${m.batch_size}, ` +
    `learning rate ${m.learning_rate}, weight decay ${m.weight_decay}${seed}.${cities}`;
  $("#export-info").textContent = `City maps exported on ${m.exported} from checkpoint ${m.checkpoint}.`;
}

function selectCity(c) {
  city = c;
  for (const button of $("#city-tabs").children) button.setAttribute("aria-selected", button.textContent === c.name);
  for (const frame of frames) {
    frame.style.aspectRatio = `${c.width} / ${c.height}`;
    images[frame.dataset.key].alt = `${c.name}: ${frame.dataset.title}`;
    images[frame.dataset.key].src = c.images[frame.dataset.key];
  }
  resetView();
  showScores(c);
}

// Read the class of every pixel back from the colors of the label and prediction maps
for (const img of [images.label, images.pred]) {
  img.addEventListener("load", () => {
    if (classIds[img.src]) return;
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(img, 0, 0);
    const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const ids = new Uint8Array(canvas.width * canvas.height);
    for (let i = 0; i < ids.length; i++) {
      ids[i] = colorIds.get((rgba[4 * i] << 16) | (rgba[4 * i + 1] << 8) | rgba[4 * i + 2]) ?? 0;
    }
    classIds[img.src] = ids;
  });
}

function showScores(c) {
  const test = c.splits.test;
  $("#scores-title").textContent = `Scores for ${c.name}`;
  $("#stat-accuracy").textContent = test ? percent(test.accuracy) : "n/a";
  $("#stat-iou").textContent = test ? test.mean_iou.toFixed(3) : "n/a";
  $("#stat-patches").textContent = c.patches.toLocaleString("en");
  // 10 m pixels
  $("#stat-size").textContent = `${(c.width / 100).toFixed(1)} × ${(c.height / 100).toFixed(1)}`;

  const splits = ["train", "validation", "test"].filter((s) => c.splits[s]);
  fillTable("#split-table", splits.map((s) => [s[0].toUpperCase() + s.slice(1),
    c.splits[s].pixels.toLocaleString("en"), percent(c.splits[s].accuracy), c.splits[s].mean_iou.toFixed(3)]));
  fillTable("#class-table", c.test_classes.map((k) => [swatch(k.id), k.pixels.toLocaleString("en"),
    k.precision.toFixed(3), k.recall.toFixed(3), k.f1.toFixed(3), k.iou.toFixed(3)]));
}

function percent(value) {
  return `${(100 * value).toFixed(1)} %`;
}

function fillTable(selector, rows) {
  const tbody = $(`${selector} tbody`);
  tbody.replaceChildren();
  for (const cells of rows) {
    const tr = tbody.insertRow();
    for (const cell of cells) tr.insertCell().append(cell);
  }
}

// Zoom and pan, the map always covers the whole frame
function applyView() {
  view.x = Math.min(0, Math.max(1 - view.s, view.x));
  view.y = Math.min(0, Math.max(1 - view.s, view.y));
  const transform = `translate(${view.x * 100}%, ${view.y * 100}%) scale(${view.s})`;
  for (const img of Object.values(images)) img.style.transform = transform;
  panels.classList.toggle("zoomed", view.s > 1);
  $("#zoom-level").textContent = `${view.s.toFixed(1)}×`;
}

function zoomAt(u, v, factor) {
  const s = Math.min(MAX_SCALE, Math.max(1, view.s * factor));
  view.x = u - (u - view.x) * s / view.s;
  view.y = v - (v - view.y) * s / view.s;
  view.s = s;
  applyView();
}

function resetView() {
  Object.assign(view, { s: 1, x: 0, y: 0 });
  applyView();
}

// Cursor position as a fraction of the frame
function framePosition(event, frame) {
  const rect = frame.getBoundingClientRect();
  return [(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height];
}

panels.addEventListener("wheel", (event) => {
  const frame = event.target.closest(".frame");
  if (!frame) return;
  // deltaMode 1 (Firefox) counts lines, about 33 px each
  const factor = Math.exp(-(event.deltaMode === 1 ? 33 : 1) * event.deltaY * 0.002);
  // At the zoom limits the wheel scrolls the page
  if ((factor < 1 && view.s === 1) || (factor > 1 && view.s === MAX_SCALE)) return;
  event.preventDefault();
  zoomAt(...framePosition(event, frame), factor);
}, { passive: false });

panels.addEventListener("pointerdown", (event) => {
  const frame = event.target.closest(".frame");
  if (!frame || event.button !== 0) return;
  drag = { frame, x: event.clientX, y: event.clientY };
  frame.setPointerCapture(event.pointerId);
  panels.classList.add("dragging");
  showCursor(event, frame);
});

panels.addEventListener("pointermove", (event) => {
  if (drag) {
    const rect = drag.frame.getBoundingClientRect();
    view.x += (event.clientX - drag.x) / rect.width;
    view.y += (event.clientY - drag.y) / rect.height;
    Object.assign(drag, { x: event.clientX, y: event.clientY });
    applyView();
  }
  const frame = event.target.closest(".frame");
  if (frame) showCursor(event, frame);
});

for (const type of ["pointerup", "pointercancel"]) {
  panels.addEventListener(type, () => {
    drag = null;
    panels.classList.remove("dragging");
  });
}
// On touch screens the tooltip stays after lifting the finger
panels.addEventListener("pointerleave", (event) => { if (event.pointerType !== "touch") hideCursor(); });
panels.addEventListener("dblclick", (event) => { if (event.target.closest(".frame")) resetView(); });
window.addEventListener("scroll", hideCursor, { passive: true });
$("#zoom-in").addEventListener("click", () => zoomAt(0.5, 0.5, 2));
$("#zoom-out").addEventListener("click", () => zoomAt(0.5, 0.5, 0.5));
$("#zoom-reset").addEventListener("click", resetView);

// Synced crosshair and the classes under the cursor
function showCursor(event, frame) {
  const [u, v] = framePosition(event, frame);
  if (u < 0 || u >= 1 || v < 0 || v >= 1) return hideCursor();
  panels.style.setProperty("--cx", `${u * 100}%`);
  panels.style.setProperty("--cy", `${v * 100}%`);
  panels.classList.add("hover");

  const pixel = Math.floor((v - view.y) / view.s * city.height) * city.width
    + Math.floor((u - view.x) / view.s * city.width);
  [images.label, images.pred].forEach((img, i) => {
    const ids = classIds[img.src];
    tooltip.children[i].lastElementChild.replaceChildren(ids ? swatch(ids[pixel]) : "loading");
  });
  tooltip.hidden = false;
  const x = event.clientX + 16 + tooltip.offsetWidth < window.innerWidth
    ? event.clientX + 16 : event.clientX - 16 - tooltip.offsetWidth;
  tooltip.style.left = `${Math.max(4, x)}px`;
  tooltip.style.top = `${Math.max(4, Math.min(event.clientY + 16, window.innerHeight - tooltip.offsetHeight - 4))}px`;
}

function hideCursor() {
  panels.classList.remove("hover");
  tooltip.hidden = true;
}
