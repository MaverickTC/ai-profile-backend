const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const sharp = require('sharp');
require('dotenv').config();
const Anthropic = require('@anthropic-ai/sdk');
const multer = require('multer');

// Replace OpenAI with Anthropic/Claude
const CLAUDE_API_KEY = process.env.CLAUDE_API_KEY;
const anthropic = new Anthropic({
  apiKey: CLAUDE_API_KEY,
});

const WEIGHTS = {
  quality: 0.30,
  aesthetics: 0.30,
  smileProb: 0.15,
  gazeBonus: 0.10,
  redFlag: 0.25,
  petFlag: 0.05,
  filterPenalty: 0.10,
  groupPenalty: 0.05,
};

const norm = (x, min, max) => (x - min) / (max - min);

async function extractFeatures(buffer) {
  const resizedBuffer = await sharp(buffer)
    .resize({ width: 512 })
    .png({ quality: 80 })
    .toBuffer();

  const base64Image = resizedBuffer.toString('base64');
  const systemPrompt = `You are a dating profile photo evaluator. Extract the following features from the image:

1. quality (40–100)
2. aesthetics (3–10)
3. smileProb (-0.5 – 1) - if face not visible value is 0.6
4. gazeDeg (0–90)
5. redFlag (true/false)
6. petFlag (true/false)
7. filterStrength (0–1)
8. numFaces (int)

Then provide a detailed assessment of the image as a dating profile picture. Return a JSON object with fields: "features" and "assessment".`;

  const response = await anthropic.messages.create({
    model: "claude-3-7-sonnet-20250219",
    max_tokens: 1000,
    system: systemPrompt,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Analyze this dating profile photo." },
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/png",
              data: base64Image
            }
          }
        ]
      }
    ],
    temperature: 0.2
  });

  const content = response.content[0]?.text;
  const jsonMatch = content.match(/```json\s*([\s\S]*?)```/) || content.match(/{[\s\S]*}/);

  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[1] || jsonMatch[0]);
    } catch (e) {
      throw new Error("Invalid JSON from Claude response.");
    }
  } else {
    throw new Error("No JSON found in Claude response.");
  }
}

function compositeScore(featuresData) {
  const f = featuresData.features;

  let gazeScore = 0;
  if (f.gazeDeg <= 15) gazeScore = 1;
  else if (f.gazeDeg <= 45) gazeScore = 0.6;
  else gazeScore = 0.3;

  const positiveScore = (
    WEIGHTS.quality * norm(f.quality, 40, 100) +
    WEIGHTS.aesthetics * norm(f.aesthetics, 3, 10) +
    WEIGHTS.smileProb * f.smileProb +
    WEIGHTS.gazeBonus * gazeScore +
    WEIGHTS.petFlag * (f.petFlag ? 1 : 0)
  );

  const penalties = (
    WEIGHTS.redFlag * (f.redFlag ? 1 : 0) +
    WEIGHTS.filterPenalty * f.filterStrength +
    WEIGHTS.groupPenalty * (f.numFaces > 3 ? 1 : 0)
  );

  const rawScore = positiveScore - penalties;
  const adjustedScore = 0.9 * rawScore + 0.2;

  return Math.max(0.2, Math.min(1, adjustedScore));
}

async function generateFeedback(featuresData, score, imageBuffer) {
  // Resize and compress the image more aggressively
  const resizedBuffer = await sharp(imageBuffer)
    .resize({ width: 512 })
    .png({ quality: 80 })
    .toBuffer();
    
  const base64Image = resizedBuffer.toString('base64');
  
  // Check image size
  const imageSizeInMB = Buffer.byteLength(base64Image) / (1024 * 1024);
  if (imageSizeInMB > 3.5) {
    throw new Error(`Image too large: ${imageSizeInMB.toFixed(2)}MB. Must be under 3.5MB.`);
  }

  //const features = featuresData.features;
  const assessment = featuresData.assessment;

  const systemPrompt = `You are a concise, kind dating photo coach.
The photo has a score of ${score}/100.

AI's assessment:
${assessment}

If this is already a strong photo (score > 75), focus on what makes it effective with 1-2 genuine compliments and small feedback for better version of THIS PHOTO.
Only provide improvement suggestions if truly needed.
If face is not visible but context makes it hard to follow the rules such as looking at camera, making face more visible, etc., then do not penalize and do not complain or suggest.

Based on this SINGLE photo only:
- Do NOT suggest adding more photos or diversity
- Focus ONLY on what's visible in THIS image
- Start each point with emojis like ✅, ❌, or 💡
- Be honest - if the photo is already good, say so instead of inventing problems
- Maximum 3 points total, each point must be one or two short sentence`;

  try {
    const response = await anthropic.messages.create({
      model: "claude-3-7-sonnet-20250219",
      max_tokens: 250,
      system: systemPrompt,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Give honest feedback for this photo for dating profile."
            },
            {
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: base64Image
              }
            }
          ]
        }
      ],
      temperature: 0.7
    });

    const content = response.content[0].text;
    const tips = content.trim().split(/\n+/);
    return tips.length > 0 ? tips : ["✅ Great photo! No changes needed."];
  } catch (error) {
    console.error("Error calling Claude API:", error.response?.data || error.message);
    throw error;
  }
}

const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit per file
});

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json());

// Fix the field name to match what's coming from the client
app.post('/analyze', upload.any(), async (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ error: 'No images uploaded' });
  }

  try {
    const results = [];

    for (const file of req.files) {
      try {
        const buffer = file.buffer;
        
        const featData = await extractFeatures(buffer); 
        const score = Math.round(compositeScore(featData) * 100);
        const feedbackLines = await generateFeedback(featData, score, buffer);

        results.push({ 
          score, 
          feedbackLines, 
          features: featData.features,
          assessment: featData.assessment 
        });
      } catch (imageError) {
        // Handle individual image errors
        console.error(`Error processing image: ${imageError.message}`);
        results.push({
          score: 0,
          feedbackLines: [`❌ Error: ${imageError.message}`],
          features: {},
          assessment: "Could not analyze this image."
        });
      }
    }

    const order = results
      .map((r, i) => ({ i, score: r.score }))
      .sort((a, b) => b.score - a.score)
      .map(o => o.i);

    const scores = results.map(r => r.score);
    const feedback = results.map(r => r.feedbackLines);
    const features = results.map(r => r.features);
    const assessments = results.map(r => r.assessment);

    res.json({
      version: "v1.0",
      scores,
      feedback,
      order,
      features,
      assessments
    });

  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ 
      error: 'analysis-failed', 
      details: err.message
    });
  }
});

app.listen(PORT, () => {
  console.log(`✅ Dating Photo Coach running on http://localhost:${PORT}`);
});
