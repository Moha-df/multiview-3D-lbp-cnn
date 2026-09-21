// Telechargement d'un sous-ensemble categorise du dataset "Google Scanned
// Objects" (objets reels scannes, maillage + texture) via l'API REST de
// Gazebo Fuel, avec un split train/test par categorie.
//
// Usage : node download_dataset.js

const fs = require("fs");
const path = require("path");

const OWNER = "GoogleResearch";
const API_BASE = `https://fuel.gazebosim.org/1.0/${OWNER}/models`;
const DATA_DIR = path.join(__dirname, "data", "raw");
const MANIFEST_PATH = path.join(__dirname, "manifest.csv");
const SEED = 42;

// Categorie Fuel -> { dossier local, nb d'exemplaires train, nb d'exemplaires test }
// Les quotas sont bornes par le nombre reel d'objets disponibles par categorie.
const CATEGORIES = {
  "Shoe": { dir: "shoe", train: 25, test: 8 },
  "Bottles and Cans and Cups": { dir: "bottles_and_cans_and_cups", train: 25, test: 8 },
  "Bag": { dir: "bag", train: 20, test: 8 },
  "Board Games": { dir: "board_games", train: 12, test: 5 },
  "Action Figures": { dir: "action_figures", train: 12, test: 5 },
};

// PRNG deterministe (mulberry32) pour un split reproductible d'une execution a l'autre.
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(array, rng) {
  const a = array.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

async function fetchAllModels() {
  const perPage = 100;
  let page = 1;
  let all = [];
  for (;;) {
    const res = await fetch(`${API_BASE}?per_page=${perPage}&page=${page}`);
    if (res.status === 404) break; // page au-dela de la derniere : fin de la pagination
    if (!res.ok) throw new Error(`Echec listing page ${page}: HTTP ${res.status}`);
    const batch = await res.json();
    if (batch.length === 0) break;
    all = all.concat(batch);
    page++;
  }
  return all;
}

async function downloadModel(name, destZip) {
  const url = `${API_BASE}/${encodeURIComponent(name)}.zip`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Echec telechargement ${name}: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  fs.writeFileSync(destZip, buf);
  return buf.length;
}

async function main() {
  console.log("Recuperation de la liste complete des objets GoogleResearch...");
  const all = await fetchAllModels();
  console.log(`${all.length} objets listes au total.`);

  const rng = mulberry32(SEED);
  const manifestRows = ["category,split,object_name,filesize_bytes,source_url"];
  const plan = [];

  for (const [category, cfg] of Object.entries(CATEGORIES)) {
    const pool = all.filter((m) => (m.categories || []).includes(category));
    const needed = cfg.train + cfg.test;
    if (pool.length < needed) {
      console.warn(
        `Attention : "${category}" n'a que ${pool.length} objets disponibles pour ${needed} demandes.`
      );
    }
    const picked = shuffle(pool, rng).slice(0, Math.min(needed, pool.length));
    const trainSet = picked.slice(0, cfg.train);
    const testSet = picked.slice(cfg.train, cfg.train + cfg.test);

    for (const m of trainSet) plan.push({ category, cfg, split: "train", model: m });
    for (const m of testSet) plan.push({ category, cfg, split: "test", model: m });

    console.log(
      `${category}: ${trainSet.length} train / ${testSet.length} test (sur ${pool.length} disponibles)`
    );
  }

  console.log(`\nTelechargement de ${plan.length} objets...\n`);

  let done = 0;
  for (const { category, cfg, split, model } of plan) {
    const destDir = path.join(DATA_DIR, cfg.dir, split);
    fs.mkdirSync(destDir, { recursive: true });
    const destZip = path.join(destDir, `${model.name}.zip`);

    if (fs.existsSync(destZip)) {
      done++;
      console.log(`[${done}/${plan.length}] deja present: ${category}/${split}/${model.name}`);
      continue;
    }

    try {
      const bytes = await downloadModel(model.name, destZip);
      done++;
      console.log(
        `[${done}/${plan.length}] OK ${category}/${split}/${model.name} (${(bytes / 1e6).toFixed(1)} Mo)`
      );
    } catch (err) {
      console.error(`[${done + 1}/${plan.length}] ECHEC ${category}/${split}/${model.name}: ${err.message}`);
    }

    manifestRows.push(
      [category, split, model.name, model.filesize, `${API_BASE}/${encodeURIComponent(model.name)}.zip`].join(",")
    );
  }

  fs.writeFileSync(MANIFEST_PATH, manifestRows.join("\n") + "\n");
  console.log(`\nManifeste ecrit dans ${MANIFEST_PATH}`);
  console.log("Termine.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
