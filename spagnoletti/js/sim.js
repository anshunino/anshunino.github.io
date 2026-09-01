/*
 * spagnoletti - many many tiny spanish men.
 *
 * Each sprite carries a mass equal to the sum of its pixel intensities, with the
 * transparent background contributing exactly zero. Copies drift, collide, and
 * fission is driven by contact: a copy-on-copy impact splits both participants
 * into N lovely little Spaniards, under two constraints:
 *
 *   mass      each lovely little Spaniard is M/N, and since one scaled by s has
 *             mass s^2 M, the geometry is forced: s = 1/sqrt(N).
 *   momentum  they recoil isotropically in the parent's rest frame,
 *             v_i = v + u_i with the u_i evenly spaced so sum(u_i) = 0 exactly.
 *
 * Isotropic recoil costs energy, so each split draws exactly ½Mu² from an
 * internal budget U. Kinetic energy gained equals internal energy spent, so
 * KE + U is conserved to machine precision - the readouts show this live.
 *
 * The budget is quoted as a population: a ceiling of C copies from S seeds is
 * worth R = log_N(C/S) generations, and U0 = R·dQ0 runs out exactly as that
 * population is reached. Nothing divides past the ceiling, collision or not.
 *
 * Collisions come in two kinds. Copy-on-copy is an elastic disc impact, exact in
 * both momentum and kinetic energy, and it is the only thing that triggers a
 * split. A boundary is an infinite mass: a pure specular reflection that keeps
 * speed - so kinetic energy is untouched - but absorbs momentum, and never
 * causes a split. Switch the boundary to periodic and momentum is exact too.
 *
 * The sprites and their masses come from 01_prepare_assets.py, which keys the
 * matte topologically and integrates alpha*luma; the manifest is inlined here so
 * the page also runs from file://.
 */

const MANIFEST = [
  { id: "many",    title: "Many",    file: "many.png",    width: 362, height: 384, mass: 29453.82,  coverage: 0.4465 },
  { id: "many_2",  title: "Many II", file: "many_2.png",  width: 271, height: 384, mass: 23106.285, coverage: 0.6123 },
  { id: "spanish", title: "Spanish", file: "spanish.png", width: 147, height: 384, mass: 12974.402, coverage: 0.4655 },
  { id: "tiny",    title: "Tiny",    file: "tiny.png",    width: 186, height: 384, mass: 18148.771, coverage: 0.4644 },
  { id: "men",     title: "Men",     file: "men.png",     width: 152, height: 249, mass: 7533.323,  coverage: 0.6996 },
];

// Mass is normalised against this so the readouts are resolution independent.
const MASS_SCALE = 29453.82;
// Hard ceiling on live sprites, independent of the energy budget. Phones and
// tablets get a lower one: the broad phase plus a few thousand drawImage calls
// per frame will not hold 60fps on a mobile GPU.
const IS_SMALL = (window.matchMedia && matchMedia("(pointer: coarse)").matches)
  || Math.min(screen.width, screen.height) < 820;
// Ceiling the device can actually draw, which is all this is for. The energy
// budget is what limits the swarm in practice.
const DEVICE_MAX = IS_SMALL ? 1024 : 4096;
// A sprite is retired once it is smaller than this many pixels on its long edge.
const MIN_PIXELS = 2.5;
// Base on-screen height of a generation-0 sprite, as a fraction of canvas height.
const BASE_HEIGHT = 0.20;
// Below this normal approach speed a contact is treated as resting, not an
// impact - stops jitter in a crowd from driving an endless split chain.
const IMPACT_MIN = 3.0;

const state = {
  running: false,
  particles: [],
  sprites: new Map(),
  selected: new Set(["spanish"]),
  N: 4,
  cooldown: 260,     // ms a copy is immune to splitting again after a split
  recoil: 46,        // u, in px/s
  budgetExp: 8,      // energy budget, quoted as a population: 2^budgetExp copies
  R: 0,              // generations the budget pays for; derived from the above
  seeds: 14,         // starting copies per selected image
  periodic: false,   // false = reflecting walls; true = wrap at the edges
  trails: true,
  showBoxes: false,
  initialMass: 0,
  initialEnergy: 0,
  initialP: 0,
  lastFrame: 0,
  splitsDone: 0,
  collisions: 0,
};

/* ---------- sprite loading + mip pyramid ---------- */

/**
 * Pre-scale each sprite into a mip chain. Drawing a 400px source into a 6px box
 * every frame is both slow and badly filtered; drawing from the nearest mip is
 * fast and properly antialiased. It is the same idea as the offline
 * deepinv.physics.Downsampling pass, though canvas filtering is not its exact
 * bicubic kernel.
 */
function buildMips(img) {
  const mips = [];
  let w = img.width, h = img.height;
  let cur = document.createElement("canvas");
  cur.width = w; cur.height = h;
  cur.getContext("2d").drawImage(img, 0, 0);
  mips.push(cur);

  while (w > 2 && h > 2) {
    w = Math.max(1, Math.floor(w / 2));
    h = Math.max(1, Math.floor(h / 2));
    const next = document.createElement("canvas");
    next.width = w; next.height = h;
    const c = next.getContext("2d");
    c.imageSmoothingEnabled = true;
    c.imageSmoothingQuality = "high";
    c.drawImage(mips[mips.length - 1], 0, 0, w, h);
    mips.push(next);
  }
  return mips;
}

function loadSprites() {
  return Promise.all(MANIFEST.map(meta => new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => {
      const aspect = meta.width / meta.height;
      // Equal-area collision disc: r = sqrt(coverage * w * h / pi), and with
      // w = h*aspect this is just h * sqrt(coverage*aspect/pi).
      const rFactor = Math.sqrt(meta.coverage * aspect / Math.PI);
      state.sprites.set(meta.id, { meta, mips: buildMips(img), aspect, rFactor });
      res();
    };
    img.onerror = () => rej(new Error(`failed to load ${meta.file}`));
    img.src = `images/${meta.file}`;
  })));
}

/* ---------- physics ---------- */

const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d");
let W = 0, H = 0;

function resize() {
  // Cap the backing store: a 3x buffer on a large phone is millions of pixels
  // to clear every frame, and iOS enforces a hard canvas area limit.
  const dpr = Math.min(window.devicePixelRatio || 1, IS_SMALL ? 1.5 : 2);
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  W = rect.width; H = rect.height;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Rotating the device shrinks the box; anything now outside is pulled back in.
  for (const p of state.particles) {
    p.x = Math.min(Math.max(p.x, 0), W);
    p.y = Math.min(Math.max(p.y, 0), H);
  }
}

function splitCost(mass) {
  return 0.5 * mass * state.recoil * state.recoil;
}

/** The energy budget, quoted directly as a population ceiling. */
function maxCopies() {
  return Math.min(Math.pow(2, state.budgetExp), DEVICE_MAX);
}

/**
 * Generations the budget pays for. Starting from S seeds, N-way splitting
 * reaches S*N^R copies after R generations, so a ceiling of C copies is worth
 *
 *     R = log_N (C / S)
 *
 * and U0 = R * dQ0 then runs out exactly as that population is reached. The
 * count check in step() enforces the same ceiling, so the two agree instead of
 * one silently overriding the other.
 */
function budgetGenerations() {
  const c = maxCopies();
  const seedTotal = Math.max(1, Math.min(state.seeds * state.selected.size, c));
  // Rounded up: generations are discrete, so a budget of, say, 1.94 of them
  // would otherwise strand the swarm a whole generation short of the ceiling.
  // Buying the last one outright lets the population reach C, and the count
  // check in step() is what stops it there - the two limits agree.
  return Math.max(0, Math.ceil(Math.log(c / seedTotal) / Math.log(state.N)));
}

/** Collision radius: the disc with the same area as the sprite's actual matte,
 *  so a mostly-empty cutout does not collide from its transparent corners. */
function radius(p) {
  const sp = state.sprites.get(p.id);
  return p.h * sp.rFactor;
}

/** Shortest separation under the active boundary. Under periodic wrapping the
 *  minimum-image convention is what keeps contact continuous across the seam. */
function delta(ax, ay, bx, by) {
  let dx = bx - ax, dy = by - ay;
  if (state.periodic) {
    if (dx >  W / 2) dx -= W; else if (dx < -W / 2) dx += W;
    if (dy >  H / 2) dy -= H; else if (dy < -H / 2) dy += H;
  }
  return [dx, dy];
}

function applyBoundary(p) {
  if (state.periodic) {
    // Wrapping moves a copy without touching its velocity, so total momentum
    // is untouched - there is no wall to absorb any.
    p.x = ((p.x % W) + W) % W;
    p.y = ((p.y % H) + H) % H;
    return;
  }
  // The wall is an infinite mass: a pure specular reflection. Speed is
  // preserved exactly, so kinetic energy is untouched, but the wall absorbs
  // the momentum change - which is why |p| is only conserved when periodic.
  // A wall is not an impact between two copies, so it never sets `hit` and
  // therefore never causes a split.
  const r = radius(p);
  if (p.x - r < 0) { p.x = r;     p.vx = Math.abs(p.vx); }
  if (p.x + r > W) { p.x = W - r; p.vx = -Math.abs(p.vx); }
  if (p.y - r < 0) { p.y = r;     p.vy = Math.abs(p.vy); }
  if (p.y + r > H) { p.y = H - r; p.vy = -Math.abs(p.vy); }
}

/**
 * Resolve one pair as an elastic collision of two discs.
 *
 *   j = 2 m_a m_b / (m_a + m_b) * (v_rel . n)
 *
 * Equal and opposite impulses along the contact normal, so total momentum is
 * unchanged by construction, and the elastic form leaves kinetic energy
 * unchanged too. Pairs already separating are skipped, which is what stops a
 * crowded contact from pumping energy in.
 */
function collide(a, b) {
  const [dx, dy] = delta(a.x, a.y, b.x, b.y);
  const rsum = radius(a) + radius(b);
  const d2 = dx * dx + dy * dy;
  if (d2 >= rsum * rsum || d2 < 1e-9) return 0;

  const d = Math.sqrt(d2);
  const nx = dx / d, ny = dy / d;
  const vn = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
  if (vn <= 0) return 0;                       // separating already

  const j = (2 * a.mass * b.mass / (a.mass + b.mass)) * vn;
  a.vx -= (j / a.mass) * nx;  a.vy -= (j / a.mass) * ny;
  b.vx += (j / b.mass) * nx;  b.vy += (j / b.mass) * ny;

  // Push apart so they do not stick. Position only - no energy, no momentum.
  const push = (rsum - d) / 2;
  a.x -= nx * push;  a.y -= ny * push;
  b.x += nx * push;  b.y += ny * push;

  return vn;                                   // impact speed along the normal
}

/**
 * Broad phase: a uniform grid whose cell is at least the largest possible
 * contact distance, so any touching pair lands in adjacent cells and a 3x3
 * sweep is exhaustive. Without this the swarm is O(n^2) and dies at a few
 * thousand copies.
 */
function resolveCollisions() {
  const ps = state.particles;
  if (ps.length < 2) return;

  let maxR = 0;
  for (const p of ps) maxR = Math.max(maxR, radius(p));
  const cell = Math.max(2 * maxR, 6);
  const nx = Math.max(1, Math.ceil(W / cell));
  const ny = Math.max(1, Math.ceil(H / cell));

  const grid = new Map();
  for (let i = 0; i < ps.length; i++) {
    const cx = Math.min(nx - 1, Math.max(0, Math.floor(ps[i].x / cell)));
    const cy = Math.min(ny - 1, Math.max(0, Math.floor(ps[i].y / cell)));
    const key = cy * nx + cx;
    const bucket = grid.get(key);
    if (bucket) bucket.push(i); else grid.set(key, [i]);
  }

  for (const [key, bucket] of grid) {
    const cx = key % nx, cy = (key - cx) / nx;
    for (let ox = 0; ox <= 1; ox++) {
      for (let oy = (ox === 0 ? 0 : -1); oy <= 1; oy++) {
        let gx = cx + ox, gy = cy + oy;
        if (state.periodic) {
          gx = ((gx % nx) + nx) % nx;
          gy = ((gy % ny) + ny) % ny;
        } else if (gx < 0 || gx >= nx || gy < 0 || gy >= ny) continue;

        const other = grid.get(gy * nx + gx);
        if (!other) continue;
        const same = (gx === cx && gy === cy);

        for (let ii = 0; ii < bucket.length; ii++) {
          for (let jj = same ? ii + 1 : 0; jj < other.length; jj++) {
            const A = ps[bucket[ii]], B = ps[other[jj]];
            if (A === B) continue;
            const vn = collide(A, B);
            if (vn > 0) {
              state.collisions++;
              // A real impact, not a resting nudge, is what triggers fission.
              if (vn >= IMPACT_MIN) { A.hit = true; B.hit = true; }
            }
          }
        }
      }
    }
  }
}

function spawn(spriteId, x, y, angle, now) {
  const sp = state.sprites.get(spriteId);
  const mass = sp.meta.mass / MASS_SCALE;
  const speed = state.recoil * 0.55;
  // U0 = R * dQ0, with R set by the population the budget allows.
  const U = state.R * splitCost(mass);
  return {
    id: spriteId,
    x, y,
    vx: Math.cos(angle) * speed,
    vy: Math.sin(angle) * speed,
    gen: 0,
    mass,
    U,
    lastSplit: now,
    hit: false,
    h: H * BASE_HEIGHT,
  };
}

function reset() {
  const now = performance.now();
  state.particles = [];
  state.splitsDone = 0;
  state.collisions = 0;
  const ids = [...state.selected];
  if (ids.length === 0) return;

  state.R = budgetGenerations();

  // Splitting is now driven by contact, so a lone copy would never divide -
  // the swarm has to start with enough members to actually meet. Round-robin
  // over the images so that trimming to the ceiling keeps them balanced.
  const wanted = [];
  for (let k = 0; k < state.seeds; k++) for (const id of ids) wanted.push(id);

  // The budget is a population ceiling, so it bounds the starting swarm too:
  // you may not begin with more copies than you are allowed to have.
  for (const id of wanted.slice(0, maxCopies())) {
    state.particles.push(spawn(
      id,
      W * (0.12 + 0.76 * Math.random()),
      H * (0.12 + 0.76 * Math.random()),
      Math.random() * Math.PI * 2,
      now,
    ));
  }

  state.initialMass = totalMass();
  state.initialEnergy = totalEnergy();
  state.initialP = totalP();
  state.lastFrame = now;
  updateReadout();
}

function split(p, now) {
  const N = state.N;
  const cost = splitCost(p.mass);
  if (p.U < cost - 1e-12) return null;

  const kids = [];
  const spaniardMass = p.mass / N;
  const spaniardU = (p.U - cost) / N;
  // s = 1/sqrt(N) is what mass conservation demands.
  const spaniardH = p.h / Math.sqrt(N);
  const phase = Math.random() * Math.PI * 2;

  for (let i = 0; i < N; i++) {
    // Evenly spaced directions => sum(u_i) = 0 => momentum is exact.
    const th = phase + (i / N) * Math.PI * 2;
    kids.push({
      id: p.id,
      x: p.x + Math.cos(th) * spaniardH * 0.45,
      y: p.y + Math.sin(th) * spaniardH * 0.45,
      vx: p.vx + Math.cos(th) * state.recoil,
      vy: p.vy + Math.sin(th) * state.recoil,
      gen: p.gen + 1,
      mass: spaniardMass,
      U: spaniardU,
      lastSplit: now,
      hit: false,
      h: spaniardH,
    });
  }
  return kids;
}

function step(dt, now) {
  for (const p of state.particles) {
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    applyBoundary(p);
  }

  resolveCollisions();

  const next = [];
  const ceiling = maxCopies();
  // Tracked as we go: a split adds N-1 net, and the budget must not be
  // exceeded even when a collision would otherwise trigger one.
  let projected = state.particles.length;

  for (const p of state.particles) {
    const ready = p.hit && (now - p.lastSplit) >= state.cooldown;
    const roomToGrow = projected + (state.N - 1) <= ceiling;
    const bigEnough = p.h / Math.sqrt(state.N) >= MIN_PIXELS;
    p.hit = false;

    if (ready && roomToGrow && bigEnough) {
      const kids = split(p, now);
      if (kids) {
        next.push(...kids);
        projected += state.N - 1;
        state.splitsDone++;
        continue;
      }
    }
    next.push(p);
  }
  state.particles = next;
}

/* ---------- conserved quantities ---------- */

const totalMass = () => state.particles.reduce((s, p) => s + p.mass, 0);
const totalP = () => {
  let px = 0, py = 0;
  for (const p of state.particles) { px += p.mass * p.vx; py += p.mass * p.vy; }
  return Math.hypot(px, py);
};
const totalKE = () => state.particles.reduce((s, p) => s + 0.5 * p.mass * (p.vx * p.vx + p.vy * p.vy), 0);
const totalU = () => state.particles.reduce((s, p) => s + p.U, 0);
const totalEnergy = () => totalKE() + totalU();

/* ---------- rendering ---------- */

function draw() {
  if (state.trails) {
    ctx.fillStyle = "rgba(10, 12, 18, 0.42)";
    ctx.fillRect(0, 0, W, H);
  } else {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0a0c12";
    ctx.fillRect(0, 0, W, H);
  }

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  for (const p of state.particles) {
    const sp = state.sprites.get(p.id);
    const h = p.h;
    const w = h * sp.aspect;

    // Pick the mip just above the target size, so the GPU only ever downscales
    // by at most 2x - this is the antialiasing the offline ladder figure shows.
    let level = Math.floor(Math.log2(sp.meta.height / Math.max(h, 1)));
    level = Math.max(0, Math.min(sp.mips.length - 1, level));
    const mip = sp.mips[level];

    // Under wrapping a copy near an edge is partly on the far side too, so it
    // gets drawn again shifted by a full period.
    const xs = [p.x], ys = [p.y];
    if (state.periodic) {
      if (p.x < w) xs.push(p.x + W); else if (p.x > W - w) xs.push(p.x - W);
      if (p.y < h) ys.push(p.y + H); else if (p.y > H - h) ys.push(p.y - H);
    }

    for (const gx of xs) {
      for (const gy of ys) {
        ctx.drawImage(mip, gx - w / 2, gy - h / 2, w, h);
        if (state.showBoxes) {
          ctx.strokeStyle = "rgba(120, 200, 255, 0.35)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(gx, gy, radius(p), 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }
  }
}

/* ---------- readouts ---------- */

const el = id => document.getElementById(id);

function updateReadout() {
  const m = totalMass();
  const ke = totalKE();
  const u = totalU();
  const e = ke + u;
  const gens = state.particles.length
    ? Math.max(...state.particles.map(p => p.gen))
    : 0;

  el("r-count").textContent = state.particles.length.toLocaleString();
  el("r-hits").textContent = state.collisions.toLocaleString();
  el("r-p").textContent = totalP().toFixed(2);
  // Only honest to flag momentum as conserved when there is no wall to eat it.
  el("tile-p").classList.toggle("keep", state.periodic);
  el("r-p-label").textContent = state.periodic
    ? "|momentum|" : "|momentum| (wall absorbs)";
  el("r-gen").textContent = gens;
  el("r-mass").textContent = state.initialMass
    ? (m / state.initialMass * 100).toFixed(2) + "%" : "--";
  el("r-ke").textContent = ke.toFixed(1);
  el("r-u").textContent = u.toFixed(1);
  el("r-energy").textContent = state.initialEnergy
    ? (e / state.initialEnergy * 100).toFixed(2) + "%" : "--";
}

/* ---------- loop ---------- */

function frame(now) {
  const dt = Math.min((now - state.lastFrame) / 1000, 0.05);
  state.lastFrame = now;
  if (state.running) step(dt, now);
  draw();
  updateReadout();
  requestAnimationFrame(frame);
}

/* ---------- controls ---------- */

function buildPicker() {
  const wrap = el("picker");
  MANIFEST.forEach(meta => {
    const b = document.createElement("button");
    b.className = "pick" + (state.selected.has(meta.id) ? " on" : "");
    b.innerHTML =
      `<img src="images/${meta.file}" alt="${meta.title}">` +
      `<span>${meta.title}</span>` +
      `<em>m = ${(meta.mass / MASS_SCALE).toFixed(2)}</em>`;
    b.onclick = () => {
      if (state.selected.has(meta.id)) {
        if (state.selected.size === 1) return;  // keep at least one
        state.selected.delete(meta.id);
      } else {
        state.selected.add(meta.id);
      }
      b.classList.toggle("on", state.selected.has(meta.id));
      refreshLabels();
      reset();
    };
    wrap.appendChild(b);
  });
}

/** Seeds actually used, after the population ceiling trims them. */
function seedsUsed() {
  return Math.min(state.seeds * state.selected.size, maxCopies());
}

/**
 * Both labels depend on the budget, the image selection and N, so they are
 * refreshed together whenever any of those move.
 */
function refreshLabels() {
  const c = maxCopies();
  const capped = Math.pow(2, state.budgetExp) > DEVICE_MAX;
  el("s-budget-val").textContent =
    `${c.toLocaleString()} copies${capped ? " (device limit)" : ""}`;

  const used = seedsUsed();
  const asked = state.seeds * state.selected.size;
  el("s-seeds-val").textContent = used < asked
    ? `${state.seeds} per image (${used} fit)`
    : `${state.seeds} per image`;
}

function bindSlider(id, key, fmt, after) {
  const input = el(id);
  const out = el(id + "-val");
  const sync = () => {
    state[key] = parseFloat(input.value);
    out.textContent = fmt(state[key]);
    if (after) after();
  };
  input.addEventListener("input", sync);
  sync();
}

function init() {
  resize();
  let resizeTimer = 0;
  const onResize = () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 120);
  };
  window.addEventListener("resize", onResize, { passive: true });
  window.addEventListener("orientationchange", onResize, { passive: true });

  // Backgrounded tabs get no rAF anyway; stopping the clock avoids one huge
  // catch-up step when the page comes back.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) state.lastFrame = performance.now();
  });

  buildPicker();

  bindSlider("s-cooldown", "cooldown", v => v + " ms");
  bindSlider("s-seeds", "seeds", () => "", () => { refreshLabels(); reset(); });
  bindSlider("s-recoil", "recoil", v => v + " px/s", reset);
  bindSlider("s-budget", "budgetExp", () => "", () => { refreshLabels(); reset(); });

  el("s-n").addEventListener("change", e => {
    state.N = parseInt(e.target.value, 10);
    refreshLabels();
    reset();
  });

  el("s-bounds").addEventListener("change", e => {
    state.periodic = e.target.value === "periodic";
    reset();
  });

  el("t-trails").addEventListener("change", e => { state.trails = e.target.checked; });
  el("t-boxes").addEventListener("change", e => { state.showBoxes = e.target.checked; });

  el("b-play").addEventListener("click", () => {
    state.running = !state.running;
    state.lastFrame = performance.now();
    el("b-play").textContent = state.running ? "Pause" : "Play";
  });
  el("b-reset").addEventListener("click", () => {
    reset();
    if (!state.running) { state.running = true; el("b-play").textContent = "Pause"; }
  });

  reset();
  requestAnimationFrame(frame);
}

loadSprites().then(init).catch(err => {
  const b = document.getElementById("sim-error");
  b.textContent = err.message + " - the sprites live in images/ next to this page.";
  b.style.display = "block";
});
