/*
  Dating‑Photo Coach — full pipeline (Node.js)
  ----------------------------------------------------
  ✓  Technical / aesthetic scoring (pluggable CV models)
  ✓  Composite ranking logic (research‑based weights)
  ✓  Natural‑language feedback via OpenAI Chat API
  ----------------------------------------------------
  Install prerequisites:
    npm install express cors body-parser sharp openai dotenv
    # + heavy CV deps if you want on‑device scoring later:
    # npm install @tensorflow/tfjs-node @vladmandic/face-api @tensorflow-models/mobilenet

  Environment vars required (see .env.example):
    OPENAI_API_KEY=sk-...
*/

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const sharp = require('sharp');
require('dotenv').config();
const { OpenAI } = require('openai');

// ────────────────────────────────────────────────────────────────────────────────
// OpenAI client
// ────────────────────────────────────────────────────────────────────────────────
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ────────────────────────────────────────────────────────────────────────────────
// Configurable research‑based weights (tweak w/ Bayesian optimisation later)
// ────────────────────────────────────────────────────────────────────────────────
const WEIGHTS = {
  quality: 0.25,
  aesthetics: 0.25,
  smileProb: 0.15,
  gazeBonus: 0.10,
  redFlag: 0.05,
  petFlag: 0.05,
  filterPenalty: 0.10,
  groupPenalty: 0.05,
};

// ────────────────────────────────────────────────────────────────────────────────
// Utility: linear normalisation helper
// ────────────────────────────────────────────────────────────────────────────────
const norm = (x, min, max) => (x - min) / (max - min);

// ────────────────────────────────────────────────────────────────────────────────
// ⚠️  Placeholder CV feature extractors
// Replace these with real models as you integrate TensorFlow / ONNX
// ────────────────────────────────────────────────────────────────────────────────
async function extractFeatures(buffer) {
  // TODO: implement real CV logic
  // For demo we stub plausible deterministic pseudo‑scores
  const hash = [...buffer.slice(0, 16)].reduce((acc, b) => acc + b, 0);
  const rand = (seed, range = 1) => ((Math.sin(seed) + 1) / 2) * range;

  const quality = 40 + rand(hash, 60);           // 40‑100
  const aesthetics = 3 + rand(hash * 1.3, 7);    // 3‑10
  const smileProb = rand(hash * 2.1);            // 0‑1
  const gazeDeg = rand(hash * 3.7, 90);          // 0‑90°
  const redFlag = rand(hash * 4.2) > 0.7;        // bool
  const petFlag = rand(hash * 5.4) > 0.8;        // bool
  const filterStrength = rand(hash * 6.0, 1);    // 0‑1
  const numFaces = Math.floor(rand(hash * 7.7, 4)); // 0‑3

  return {
    quality,
    aesthetics,
    smileProb,
    gazeDeg,
    redFlag,
    petFlag,
    filterStrength,
    numFaces,
  };
}

// ────────────────────────────────────────────────────────────────────────────────
// Composite score according to research‑based formula
// ────────────────────────────────────────────────────────────────────────────────
function compositeScore(f) {
  return (
    WEIGHTS.quality * norm(f.quality, 40, 100) +
    WEIGHTS.aesthetics * norm(f.aesthetics, 3, 10) +
    WEIGHTS.smileProb * f.smileProb +
    WEIGHTS.gazeBonus * (1 - f.gazeDeg / 90) +
    WEIGHTS.redFlag * (f.redFlag ? 1 : 0) +
    WEIGHTS.petFlag * (f.petFlag ? 1 : 0) -
    WEIGHTS.filterPenalty * f.filterStrength -
    WEIGHTS.groupPenalty * (f.numFaces > 3 ? 1 : 0)
  ); // result ∈ roughly 0‑1
}

// ────────────────────────────────────────────────────────────────────────────────
// Feedback generation — powered by GPT‑4o
// ────────────────────────────────────────────────────────────────────────────────
async function generateFeedback(feature, score) {
  const systemPrompt = `You are a concise dating‑photo coach. You know the evidence on what makes a successful dating‑app photo (lighting, smile, solo vs group, etc.). Give 2‑4 actionable tips, each starting with an emoji (✅, ❌, 💡…), based on the provided feature data. If the overall score is low you can be gently critical.`;

  const userPrompt = JSON.stringify({ feature, score }, null, 2);

  const chatResp = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    temperature: 0.7,
    max_tokens: 120,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
  });

  return chatResp.choices[0].message.content.trim().split(/\n+/);
}

// ────────────────────────────────────────────────────────────────────────────────
// Express server
// ────────────────────────────────────────────────────────────────────────────────
const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json({ limit: '15mb' })); // handle base64 images

// POST /analyze  { images: [ 'data:image/jpeg;base64,...', ... ] }
app.post('/analyze', async (req, res) => {
  const { images } = req.body;
  if (!Array.isArray(images) || images.length === 0) {
    return res.status(400).json({ error: 'images must be a non‑empty array' });
  }

  try {
    const results = [];

    // Process each image sequentially (or Promise.all for parallel)
    for (const b64 of images) {
      // Strip data‑URI prefix if present
      const base64Data = b64.replace(/^data:image\/\w+;base64,/, '');
      const buffer = Buffer.from(base64Data, 'base64');

      // Optional: downscale large images to speed‑up CV
      const resizedBuffer = await sharp(buffer).resize({ width: 640 }).toBuffer();

      // --- CV feature extraction
      const feat = await extractFeatures(resizedBuffer);

      // --- Composite score (0‑100)
      const score = Math.round(compositeScore(feat) * 100);

      // --- GPT‑based feedback lines
      const feedbackLines = await generateFeedback(feat, score);

      results.push({ score, feedbackLines });
    }

    // Order indices by descending score
    const order = results
      .map((r, i) => ({ i, score: r.score }))
      .sort((a, b) => b.score - a.score)
      .map((o) => o.i);

    // Split arrays for backwards compatibility
    const scores = results.map((r) => r.score);
    const feedback = results.map((r) => r.feedbackLines);

    res.json({ scores, feedback, order });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'analysis‑failed', details: err.message });
  }
});

app.listen(PORT, () => console.log(`Dating‑photo coach running on http://localhost:${PORT}`));
