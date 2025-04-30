const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const sharp = require('sharp');
require('dotenv').config();
const { GoogleGenerativeAI } = require('@google/generative-ai');
const multer = require('multer');

// Replace Anthropic/Claude with Google Gemini
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

// Remove BASE_WEIGHTS and use only context-specific weights
// Each context has weights that sum to 1
const CONTEXT_OVERRIDES = {
  primary_headshot: { 
    quality: 0.15,
    aesthetics: 0.24,
    smileProb: 0.12,
    gazeBonus: 0.09,
    redFlag: 0.15,
    petFlag: 0,
    filterPenalty: 0.06,
    groupPenalty: 0.09,
    postureBonus: 0.10
  },
  hobby_activity: { 
    quality: 0.14,
    aesthetics: 0.47,
    smileProb: 0.05,
    gazeBonus: 0.05,
    redFlag: 0.14,
    petFlag: 0.02,
    filterPenalty: 0.05,
    groupPenalty: 0.03,
    postureBonus: 0.05
  },
  full_body: { 
    quality: 0.15,
    aesthetics: 0.26,
    smileProb: 0.07,
    gazeBonus: 0.06,
    redFlag: 0.15,
    petFlag: 0.01,
    filterPenalty: 0.07,
    groupPenalty: 0.04,
    postureBonus: 0.19
  },
  pet: { 
    quality: 0.15,
    aesthetics: 0.25,
    smileProb: 0.10,
    gazeBonus: 0.05,
    redFlag: 0.15,
    petFlag: 0.12,
    filterPenalty: 0.05,
    groupPenalty: 0.03,
    postureBonus: 0.10
  },
  group_social: { 
    quality: 0.15,
    aesthetics: 0.30,
    smileProb: 0.20,
    gazeBonus: 0.05,
    redFlag: 0.15,
    petFlag: 0.02,
    filterPenalty: 0.08,
    groupPenalty: 0,
    postureBonus: 0.05
  },
  generic: { 
    quality: 0.15,
    aesthetics: 0.22,
    smileProb: 0.11,
    gazeBonus: 0.07,
    redFlag: 0.15,
    petFlag: 0.04,
    filterPenalty: 0.06,
    groupPenalty: 0.05,
    postureBonus: 0.15
  }
};

// Desired slots in priority order
const SLOT_ORDER = [
  "primary_headshot",
  "full_body",
  "hobby_activity",
  "pet",
  "group_social"
];

const norm = (x, min, max) => (x - min) / (max - min);

async function extractFeatures(buffer) {
  const resizedBuffer = await sharp(buffer)
    .resize({ width: 512 })
    .png({ quality: 80 })
    .toBuffer();

  const base64Image = resizedBuffer.toString('base64');
  const systemPrompt = `You are a dating profile photo evaluator with expertise in what makes an ideal dating profile image. Extract the following features from the image:

1. quality (0 - 100) – visual clarity and resolution
2. aesthetics (0 - 100) – how attractive and appealing the person looks
3. smileProb (-50 – 120) – how warm and inviting the smile is (negative if not smiling, 60 if not visible)
4. gazeDeg (0 – 90) – angle of gaze from direct camera
5. redFlag (true/false) – any potential red flags
6. petFlag (true/false) – visible friendly animal (e.g. dog or cat)
7. filterStrength (0–1) – how edited or artificial the image looks
8. numFaces (int) – total number of visible people
9. postureScore (-1 – 1) – open, confident, inviting body language (-1 = worst, 1 = best)

Evaluation principles:
- Facial expressions: Genuine smiles with eye contact are effective and trustworthy
- Body language: Open, expansive postures appear more confident and attractive
- Clothing/grooming: Well-fitting, clean clothes and good grooming signal self-respect
- Activities: Photos showing hobbies or interests create conversation starters
- Pet photos: Including pets (especially dogs) often increases engagement
- Photo quality: Clear, well-lit photos without heavy filters perform best
- Cultural context: Consider cultural norms regarding modesty and presentation

CRITICAL RULES:
- NEVER suggest adding different photos or replacing this photo with another one
- DO NOT say things like "consider including a photo where your face is visible" or similar suggestions
- If the photo has limitations (like face not visible), but works well for what it is (e.g., showing a hobby), focus on its positive aspects

Based on this SINGLE photo:
- Do NOT mention other photos, placemnent of photos, order of photo, do not tell first, seconary, third photo etc.

Return JSON with 3 top-level keys:
  features      – existing numeric/boolean fields
  assessment    – free-text paragraph
  photoType     – one of:
      "primary_headshot"
      "full_body"
      "hobby_activity"
      "pet"
      "group_social"
      "generic" (if none of the above clearly apply)`;

  // Initialize Gemini model
  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-lite" });

  // Create parts for the request
  const imagePart = {
    inlineData: {
      data: base64Image,
      mimeType: "image/png"
    }
  };

  try {
    const result = await model.generateContent({
      contents: [{ 
        role: "user", 
        parts: [
          { text: systemPrompt },
          { text: "Analyze this dating profile photo." },
          imagePart
        ]
      }],
      generationConfig: {
        temperature: 0.2,
        maxOutputTokens: 1000,
      }
    });

    const response = result.response;
    const content = response.text();
    const jsonMatch = content.match(/```json\s*([\s\S]*?)```/) || content.match(/{[\s\S]*}/);

    if (jsonMatch) {
      try {
        const parsed = JSON.parse(jsonMatch[1] || jsonMatch[0]);
        return {
          features: parsed.features,
          assessment: parsed.assessment,
          photoType: parsed.photoType || "generic"
        };
      } catch (e) {
        throw new Error("Invalid JSON from Gemini response.");
      }
    } else {
      throw new Error("No JSON found in Gemini response.");
    }
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    throw error;
  }
}

function compositeScore(data) {
  const f = data.features;
  const type = data.photoType;
  
  // Use context-specific weights directly without merging with BASE_WEIGHTS
  const W = CONTEXT_OVERRIDES[type] || CONTEXT_OVERRIDES.generic;

  let gazeScore = 0;
  if (f.gazeDeg <= 15) gazeScore = 1;
  else if (f.gazeDeg <= 45) gazeScore = 0.6;
  else gazeScore = 0.3;

  // Calculate normalized values for logging
  const normalizedQuality = norm(f.quality, 0, 100);
  const normalizedAesthetics = norm(f.aesthetics, 0, 100);
  const normalizedSmileProb = norm(f.smileProb, -50, 100);
  
  // Add penalties for not smiling and not looking at camera
  const smilePenalty = f.smileProb < 0 ? 0.1 : 0;
  
  // Adjust gaze penalty based on photo type
  let gazePenalty = 0;
  if (type !== "hobby_activity" && type !== "pet") {
    // Standard gaze penalty for most photo types
    gazePenalty = f.gazeDeg > 30 ? 0.1 : 0;
    
    // Add extra combined penalty if both not smiling AND not looking at camera
    if (f.smileProb < 0 && f.gazeDeg > 30) {
      gazePenalty += 0.15; // Heavier penalty for the combination
    }
  } else {
    // Reduced gaze penalty for hobby and pet photos
    gazePenalty = f.gazeDeg > 45 ? 0.05 : 0;
  }
  
  const positiveScore = (
    W.quality * normalizedQuality +
    W.aesthetics * normalizedAesthetics +
    W.smileProb * normalizedSmileProb +
    W.gazeBonus * gazeScore +
    W.petFlag * (f.petFlag ? 1 : 0) +
    W.postureBonus * (f.postureScore ?? 0)
  );

  const penalties = (
    W.redFlag * (f.redFlag ? 1 : 0) +
    W.filterPenalty * f.filterStrength +
    // NOTE – group penalty now only if >2 faces *and* context isn't group_social
    W.groupPenalty * ((type !== "group_social" && f.numFaces > 2) ? 1 : 0) +
    smilePenalty +
    gazePenalty
  );

  const rawScore = positiveScore - penalties;
  const adjustedScore = rawScore + 0.2;
  const finalScore = Math.max(0.2, Math.min(1, adjustedScore));

  // Print all scoring details
  console.log('\n===== SCORING DETAILS =====');
  console.log('PHOTO TYPE:', type);
  console.log('WEIGHTS:', JSON.stringify(W, null, 2));
  console.log('FEATURES:', JSON.stringify(f, null, 2));
  
  console.log('\nNORMALIZED VALUES:');
  console.log(`- Quality: ${f.quality} → ${normalizedQuality.toFixed(2)}`);
  console.log(`- Aesthetics: ${f.aesthetics} → ${normalizedAesthetics.toFixed(2)}`);
  console.log(`- Gaze: ${f.gazeDeg}° → ${gazeScore.toFixed(2)}`);
  
  console.log('\nPOSITIVE CONTRIBUTIONS:');
  console.log(`- Quality: ${W.quality} × ${normalizedQuality.toFixed(2)} = ${(W.quality * normalizedQuality).toFixed(2)}`);
  console.log(`- Aesthetics: ${W.aesthetics} × ${normalizedAesthetics.toFixed(2)} = ${(W.aesthetics * normalizedAesthetics).toFixed(2)}`);
  console.log(`- Smile: ${W.smileProb} × ${f.smileProb} = ${(W.smileProb * f.smileProb).toFixed(2)}`);
  console.log(`- Gaze: ${W.gazeBonus} × ${gazeScore} = ${(W.gazeBonus * gazeScore).toFixed(2)}`);
  console.log(`- Pet: ${W.petFlag} × ${f.petFlag ? 1 : 0} = ${(W.petFlag * (f.petFlag ? 1 : 0)).toFixed(2)}`);
  console.log(`- Posture: ${W.postureBonus} × ${f.postureScore ?? 0} = ${(W.postureBonus * (f.postureScore ?? 0)).toFixed(2)}`);
  console.log(`- Total Positive: ${positiveScore.toFixed(2)}`);
  
  console.log('\nPENALTIES:');
  console.log(`- Red Flag: ${W.redFlag} × ${f.redFlag ? 1 : 0} = ${(W.redFlag * (f.redFlag ? 1 : 0)).toFixed(2)}`);
  console.log(`- Filter: ${W.filterPenalty} × ${f.filterStrength} = ${(W.filterPenalty * f.filterStrength).toFixed(2)}`);
  console.log(`- Group: ${W.groupPenalty} × ${(type !== "group_social" && f.numFaces > 2) ? 1 : 0} = ${(W.groupPenalty * ((type !== "group_social" && f.numFaces > 2) ? 1 : 0)).toFixed(2)}`);
  console.log(`- Not Smiling: ${smilePenalty.toFixed(2)}`);
  console.log(`- Not Looking at Camera: ${gazePenalty.toFixed(2)}`);
  console.log(`- Total Penalties: ${penalties.toFixed(2)}`);
  
  console.log('\nFINAL CALCULATION:');
  console.log(`- Raw Score: ${positiveScore.toFixed(2)} - ${penalties.toFixed(2)} = ${rawScore.toFixed(2)}`);
  console.log(`- Adjusted Score: 0.9 × ${rawScore.toFixed(2)} + 0.2 = ${adjustedScore.toFixed(2)}`);
  console.log(`- Final Score (clamped): ${finalScore.toFixed(2)} (${Math.round(finalScore * 100)}/100)`);
  console.log('============================\n');

  return finalScore;
}

async function generateFeedback(featuresData, score, imageBuffer) {
  const resizedBuffer = await sharp(imageBuffer)
    .resize({ width: 512 })
    .png({ quality: 80 })
    .toBuffer();

  const base64Image = resizedBuffer.toString('base64');
  const imageSizeInMB = Buffer.byteLength(base64Image) / (1024 * 1024);
  if (imageSizeInMB > 3.5) {
    throw new Error(`Image too large: ${imageSizeInMB.toFixed(2)}MB. Must be under 3.5MB.`);
  }

  const assessment = featuresData.assessment;

  const systemPrompt = `You are a concise, kind dating photo coach with expertise in what makes an ideal dating profile image.
The photo has a score of ${score}/100.

AI's assessment:
${assessment}

Evaluation principles:
- Facial expressions: Genuine smiles with eye contact are effective and trustworthy
- Body language: Open, expansive postures appear more confident and attractive
- Clothing/grooming: Well-fitting, clean clothes and good grooming signal self-respect
- Activities: Photos showing hobbies or interests create conversation starters
- Pet photos: Including pets (especially dogs) often increases engagement
- Photo quality: Clear, well-lit photos without heavy filters perform best
- Cultural context: Consider cultural norms regarding modesty and presentation

IMPORTANT: Evaluate this as just ONE of the photos in a dating profile.
If this appears to be an activity/hobby photo (sports, racing, etc.), focus on how it shows personality and interests.

CRITICAL RULES:
- NEVER suggest adding different photos or replacing this photo with another one
- If face is not visible, DO NOT suggest showing face in another photo
- DO NOT say things like "consider including a photo where your face is visible" or similar suggestions
- ONLY comment on the strengths and qualities of THIS SPECIFIC PHOTO
- If the photo has limitations (like face not visible), but works well for what it is (e.g., showing a hobby), focus on its positive aspects

RULES:
- DO NOT include any introductory text, just start with points.
- NEVER suggest adding/replacing photos
- If face not visible, DO NOT suggest showing face
- ONLY comment on THIS SPECIFIC PHOTO
- For strong photos (score > 75), focus on strengths with 1-2 compliments
- Start each point with emojis (✅, 💡, ❌)
- Maximum 3 points, each 1-2 short sentences`;

  try {
    // Initialize Gemini model
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-lite" });

    // Create parts for the request
    const imagePart = {
      inlineData: {
        data: base64Image,
        mimeType: "image/png"
      }
    };

    const result = await model.generateContent({
      contents: [{ 
        role: "user", 
        parts: [
          { text: systemPrompt },
          { text: "Give honest feedback for this photo for dating profile." },
          imagePart
        ]
      }],
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 250,
      }
    });

    const response = result.response;
    const content = response.text();
    const tips = content.trim().split(/\n+/);
    
    // Sort tips by emoji priority: ✅ first, then 💡, then ❌
    const sortedTips = tips.sort((a, b) => {
      const getEmojiPriority = (tip) => {
        if (tip.startsWith('✅')) return 1;
        if (tip.startsWith('💡')) return 2;
        if (tip.startsWith('❌')) return 3;
        return 4; // Any other emoji or no emoji
      };
      
      return getEmojiPriority(a) - getEmojiPriority(b);
    });
    
    return sortedTips.length > 0 ? sortedTips : ["✅ Great photo! No changes needed."];
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    throw error;
  }
}

const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }
});

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json());

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
          assessment: featData.assessment,
          photoType: featData.photoType
        });
      } catch (imageError) {
        console.error(`Error processing image: ${imageError.message}`);
        results.push({
          score: 0,
          feedbackLines: [`❌ Error: ${imageError.message}`],
          features: {},
          assessment: "Could not analyze this image.",
          photoType: "generic"
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
    const photoTypes = results.map(r => r.photoType);

    res.json({
      version: "v1.1",
      scores,
      feedback,
      order,
      features,
      assessments,
      photoTypes
    });

  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ 
      error: 'analysis-failed', 
      details: err.message
    });
  }
});

app.post('/optimize-profile', upload.any(), async (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ error: 'No images uploaded' });
  }

  try {
    const results = [];

    // Analyze each uploaded image
    for (const file of req.files) {
      try {
        const buffer = file.buffer;
        const featData = await extractFeatures(buffer); 
        const score = Math.round(compositeScore(featData) * 100);
        
        // Store analysis results with original file information
        results.push({ 
          score,
          features: featData.features,
          assessment: featData.assessment,
          photoType: featData.photoType,
          filename: file.originalname,
          buffer: buffer
        });
      } catch (imageError) {
        console.error(`Error processing image: ${imageError.message}`);
        results.push({
          score: 0,
          features: {},
          assessment: "Could not analyze this image.",
          photoType: "generic",
          filename: file.originalname,
          error: imageError.message
        });
      }
    }

    // --- NEW SELECTION LOGIC ---
    const selected = [];

    // 4a. guarantee each slot (best-scoring photo of that type)
    SLOT_ORDER.forEach(slot => {
      const best = results
          .filter(r => r.photoType === slot)
          .sort((a,b) => b.score - a.score)[0];
      if (best) selected.push(best);
    });

    // 4b. fill remaining slots (up to 6) with highest scores not chosen yet
    results
      .filter(r => !selected.includes(r))
      .sort((a,b) => b.score - a.score)
      .slice(0, 6 - selected.length)
      .forEach(r => selected.push(r));

    // Generate profile optimization feedback
    const profileFeedback = await generateProfileFeedback(selected);

    res.json({
      version: "v1.1",
      selectedImages: selected.map(img => ({
        filename: img.filename,
        score: img.score,
        features: img.features,
        assessment: img.assessment,
        photoType: img.photoType
      })),
      profileFeedback,
      totalImagesAnalyzed: results.length
    });

  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ 
      error: 'profile-optimization-failed', 
      details: err.message
    });
  }
});

async function generateProfileFeedback(selectedImages) {
  // Create a prompt that describes the selected images
  const imagesDescription = selectedImages.map((img, index) => {
    return `Image ${index + 1} (Score: ${img.score}/100): ${img.assessment}`;
  }).join('\n\n');

  const systemPrompt = `You are a dating profile optimization expert. Based on the following 
selected images for a dating profile, provide strategic advice on how to arrange them and 
what each image contributes to the overall profile.

SELECTED IMAGES:
${imagesDescription}

Provide feedback on:
1. The optimal order to arrange these photos
2. What each photo contributes to the profile
3. Any gaps or improvements that could be made with these specific photos

CRITICAL RULES:
- DO NOT suggest adding different photos or replacing these photos
- Focus ONLY on optimizing the arrangement and presentation of THESE specific photos
- Be concise and practical in your advice
- Start with a brief overall assessment, then provide 3-5 specific recommendations`;

  try {
    // Initialize Gemini model
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-lite" });

    const result = await model.generateContent({
      contents: [{ 
        role: "user", 
        parts: [{ text: systemPrompt }]
      }],
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 500,
      }
    });

    const response = result.response;
    return response.text().trim();
  } catch (error) {
    console.error("Error generating profile feedback:", error);
    return "Could not generate profile optimization feedback due to an error.";
  }
}

app.post('/analyze-single', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }

  try {
    const buffer = req.file.buffer;
    
    // Extract features from the image
    const featData = await extractFeatures(buffer); 
    const score = Math.round(compositeScore(featData) * 100);
    const feedbackLines = await generateFeedback(featData, score, buffer);

    res.json({
      version: "v1.1",
      score,
      feedback: feedbackLines,
      features: featData.features,
      assessment: featData.assessment,
      photoType: featData.photoType
    });

  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ 
      error: 'analysis-failed', 
      details: err.message
    });
  }
});

app.post('/optimize-selection', bodyParser.json(), async (req, res) => {
  try {
    const { photos } = req.body;
    
    if (!photos || !Array.isArray(photos) || photos.length === 0) {
      return res.status(400).json({ 
        error: 'invalid-request', 
        details: 'Photos array is required' 
      });
    }

    console.log(`Optimizing selection for ${photos.length} photos`);
    
    // Step 1: Ensure we have all required data for each photo
    const validPhotos = photos.filter(photo => 
      photo.index !== undefined && 
      photo.score !== undefined && 
      photo.photoType !== undefined
    );
    
    if (validPhotos.length === 0) {
      return res.status(400).json({ 
        error: 'invalid-data', 
        details: 'No valid photos provided' 
      });
    }

    // Step 2: Select optimal photos
    const selectedIndices = selectOptimalPhotos(validPhotos);
    
    // Step 3: Generate feedback about the selection
    const profileFeedback = generateSelectionFeedback(validPhotos, selectedIndices);
    
    // Return the results
    res.json({
      version: "v1.0",
      optimalOrder: selectedIndices,
      profileFeedback
    });

  } catch (err) {
    console.error("Error optimizing selection:", err);
    res.status(500).json({ 
      error: 'optimization-failed', 
      details: err.message
    });
  }
});

// Function to select optimal photos based on role diversity and scores
function selectOptimalPhotos(photos) {
  // Maximum number of photos to select
  const MAX_PHOTOS = 6;
  
  // Slot weights for scoring (higher weight for earlier positions)
  const SLOT_WEIGHTS = [1.3, 1.15, 1.1, 1.05, 1.0, 0.95];
  
  // Desired slots in priority order
  const SLOT_ORDER = [
    "primary_headshot",
    "full_body",
    "hobby_activity",
    "pet",
    "group_social"
  ];
  
  // Step 1: If we have 6 or fewer photos, use all of them sorted by score
  if (photos.length <= MAX_PHOTOS) {
    return photos
      .sort((a, b) => b.score - a.score)
      .map(photo => photo.index);
  }
  
  // Step 2: Select best photo for each role in priority order
  const selectedPhotos = [];
  const usedIndices = new Set();
  
  for (const role of SLOT_ORDER) {
    // Find best photo of this role that's not already selected
    const bestForRole = photos
      .filter(photo => 
        photo.photoType === role && 
        !usedIndices.has(photo.index)
      )
      .sort((a, b) => b.score - a.score)[0];
    
    if (bestForRole) {
      selectedPhotos.push(bestForRole);
      usedIndices.add(bestForRole.index);
      
      // Stop if we've selected MAX_PHOTOS
      if (selectedPhotos.length >= MAX_PHOTOS) {
        break;
      }
    }
  }
  
  // Step 3: Fill remaining slots with highest scoring photos not yet selected
  if (selectedPhotos.length < MAX_PHOTOS) {
    const remainingPhotos = photos
      .filter(photo => !usedIndices.has(photo.index))
      .sort((a, b) => b.score - a.score);
    
    for (const photo of remainingPhotos) {
      selectedPhotos.push(photo);
      usedIndices.add(photo.index);
      
      if (selectedPhotos.length >= MAX_PHOTOS) {
        break;
      }
    }
  }
  
  // Step 4: Calculate weighted scores for final ordering
  const weightedPhotos = selectedPhotos.map(photo => ({
    ...photo,
    weightedScore: photo.score * (SLOT_WEIGHTS[selectedPhotos.indexOf(photo)] || 1)
  }));
  
  // Return indices in order of weighted score
  return weightedPhotos
    .sort((a, b) => b.weightedScore - a.weightedScore)
    .map(photo => photo.index);
}

// Function to generate feedback about the photo selection
function generateSelectionFeedback(allPhotos, selectedIndices) {
  // Get the selected photos
  const selectedPhotos = selectedIndices.map(index => 
    allPhotos.find(photo => photo.index === index)
  ).filter(Boolean);
  
  // Count the number of each photo type
  const typeCounts = {};
  for (const photo of selectedPhotos) {
    typeCounts[photo.photoType] = (typeCounts[photo.photoType] || 0) + 1;
  }
  
  // Calculate the average score
  const avgScore = selectedPhotos.reduce((sum, photo) => sum + photo.score, 0) / selectedPhotos.length;
  
  // Generate feedback based on the selection
  let feedback = `Your profile has an average photo score of ${Math.round(avgScore)}/100. `;
  
  // Check for role diversity
  const uniqueRoles = Object.keys(typeCounts).length;
  if (uniqueRoles >= 4) {
    feedback += "You have excellent photo diversity showing different aspects of your life. ";
  } else if (uniqueRoles >= 3) {
    feedback += "You have good photo diversity, but could benefit from more variety. ";
  } else {
    feedback += "Your profile would benefit from more diverse photos showing different aspects of your life. ";
  }
  
  // Add specific role feedback
  if (typeCounts["primary_headshot"] >= 1) {
    feedback += "✅ You have a strong headshot which is essential. ";
  } else {
    feedback += "💡 Consider adding a clear headshot where your face is visible. ";
  }
  
  if (typeCounts["full_body"] >= 1) {
    feedback += "✅ Your full body photo helps show your style. ";
  }
  
  if (typeCounts["hobby_activity"] >= 1) {
    feedback += "✅ Activity photos show your interests and personality. ";
  }
  
  if (typeCounts["pet"] >= 1) {
    feedback += "✅ Pet photos often increase engagement. ";
  }
  
  if (typeCounts["group_social"] >= 1) {
    feedback += "✅ Social photos show you're well-connected. ";
  }
  
  // Add ordering advice
  feedback += "\n\nRecommended photo order: ";
  feedback += "Start with your strongest headshot, followed by full body and activity photos. ";
  feedback += "This order has been optimized based on both photo quality and type diversity.";
  
  return feedback;
}

app.listen(PORT, () => {
  console.log(`✅ Dating Photo Coach running on http://localhost:${PORT}`);
});