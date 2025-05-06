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

// Desired slots in priority order (can be used as guidance for the AI)
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
  const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-preview-04-17" });

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
  /*console.log('\n===== SCORING DETAILS =====');
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
  console.log('============================\n');*/

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
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-preview-04-17" });

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
    // Map each file to a promise that resolves with its analysis result or throws an error
    const analysisPromises = req.files.map(async (file) => {
      try {
        const buffer = file.buffer;
        const featData = await extractFeatures(buffer);
        const score = Math.round(compositeScore(featData) * 100);
        const feedbackLines = await generateFeedback(featData, score, buffer);

        // On success, return the data object directly
        return {
          score,
          feedbackLines,
          features: featData.features,
          assessment: featData.assessment,
          photoType: featData.photoType
        };
      } catch (imageError) {
        // Log the error and add filename context
        console.error(`Error processing image ${file.originalname}: ${imageError.message}`);
        imageError.filename = file.originalname; // Add filename for context later
        // Throw the error so Promise.allSettled catches it as 'rejected'
        throw imageError;
      }
    });

    // Wait for all analysis promises to settle
    const settledResults = await Promise.allSettled(analysisPromises);

    // Process the results from Promise.allSettled
    const results = settledResults.map((result, index) => {
      if (result.status === 'fulfilled') {
        // Success: use the resolved value
        return result.value;
      } else {
        // Failure: construct the error object using the reason
        const reason = result.reason || new Error('Unknown analysis error');
        // Attempt to get filename from error or fallback to original file array
        const filename = reason.filename || req.files?.[index]?.originalname || `image ${index + 1}`;
        // Log the specific failure reason here
        console.error(`Analysis failed for ${filename}: ${reason.message}`);
        return {
          score: null, // Use null for score on error
          feedbackLines: [`❌ Error processing ${filename}: ${reason.message}`],
          features: null,
          assessment: `Could not analyze ${filename}.`,
          photoType: "generic",
          error: reason.message // Include error message
        };
      }
    });

    // Calculate order based on scores (indices correspond to the original req.files order)
    // Handle potential null scores during sorting by treating them as lowest score
    const order = results
      .map((r, i) => ({ i, score: r.score ?? -1 })) // Use -1 for null scores for sorting
      .sort((a, b) => b.score - a.score)
      .map(o => o.i);

    // Extract data arrays for the response
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
    // Catch errors not related to individual image processing (e.g., server setup)
    console.error("Server error in /analyze:", err);
    res.status(500).json({
      error: 'analysis-failed',
      details: err.message
    });
  }
});

// NEW FUNCTION: Uses AI to select and order photos based on analysis data
async function getAISelectedOrderAndFeedback(analyzedPhotos) {
  if (!analyzedPhotos || analyzedPhotos.length === 0) {
    return {
      optimalOrder: [],
      profileFeedback: "No photos available for selection."
    };
  }

  // Prepare the input for the AI prompt
  const photoDescriptions = analyzedPhotos.map(photo => {
    // Include original index for identification
    return `Photo Index ${photo.index}:
  - Score: ${photo.score}/100
  - Type: ${photo.photoType}
  - AI Assessment: ${photo.assessment}`;
  }).join('\n\n');

  // --- MODIFIED PROMPT (Asking for String) ---
  const systemPrompt = `You are an expert dating profile curator. Your task is to select the optimal set of up to 6 photos from the following list, determine the best display order, and provide actionable improvement tips as a single string.

Consider these criteria for selection and ordering:
1.  **Overall Quality & Appeal:** Use the provided 'Score' as a primary guide. Higher scores are generally better.
2.  **Photo Type Diversity:** Aim for a good mix of photo types (e.g., headshot, full body, activity, social). Use the 'Type' field. Prioritize including at least one strong 'primary_headshot' if available. Refer to this preferred type order: ${SLOT_ORDER.join(', ')}.
3.  **Content & Narrative:** Read the 'AI Assessment' for each photo. Avoid selecting photos that are too visually similar or repetitive. Choose photos that collectively paint a well-rounded picture.
4.  **Optimal Order:** Arrange the selected photos logically (strongest headshot first, then engaging shots).

Available Photos:
${photoDescriptions}

Instructions:
- Select a maximum of 6 photos. If fewer are available, select all.
- Provide your response ONLY as a JSON object containing two keys:
  - "selected_order": An array of the original photo indices (e.g., [3, 0, 5, 1, 4, 2]) representing the chosen photos in the optimal display order.
  - "improvement_tips": A single string containing 2-4 specific, actionable tips for improving the profile based on ALL analyzed photos (selected and unselected).
      - Identify weaknesses (e.g., missing photo types, low-scoring essential photos).
      - Suggest concrete actions (e.g., "Consider replacing photo [index] (score: X) with...", "Adding a photo showing [activity/social setting] could...").
      - Reference specific photo indices when suggesting replacements.
      - Focus on constructive advice, avoid generic statements.
      - Start tips with relevant emojis (✅ for strengths, 💡 for suggestions).
      - Separate each tip with a newline character (\n).

Example JSON Output:
\`\`\`json
{
  "selected_order": [1, 4, 0, 5, 2],
  "improvement_tips": "✅ Great start with the high-scoring headshot (1) and engaging activity shot (4)!\n💡 Consider replacing photo 3 (score: 65, type: generic) with a clearer full-body shot if you have one.\n💡 Adding a group photo showing you with friends could showcase your social side."
}
\`\`\`

Provide only the JSON object in your response.`;
  // --- END OF MODIFIED PROMPT ---

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-preview-04-17" });

    const result = await model.generateContent({
      contents: [{
        role: "user",
        parts: [{ text: systemPrompt }]
      }],
      generationConfig: {
        temperature: 0.6,
        maxOutputTokens: 800,
        responseMimeType: "application/json",
      }
    });

    const response = result.response;
    const responseText = response.text();
    const parsedResult = JSON.parse(responseText);

    // --- UPDATED VALIDATION (Checking for String) ---
    if (!parsedResult.selected_order || typeof parsedResult.improvement_tips !== 'string') {
        throw new Error("AI response missing required 'selected_order' or 'improvement_tips' (as string) keys.");
    }
    if (!Array.isArray(parsedResult.selected_order) || !parsedResult.selected_order.every(n => typeof n === 'number')) {
        throw new Error("AI response 'selected_order' is not an array of numbers.");
    }
    // --- END OF UPDATED VALIDATION ---

    console.log("AI Improvement Tips (String):", parsedResult.improvement_tips);

    // --- UPDATED RETURN OBJECT (Returning String) ---
    return {
      optimalOrder: parsedResult.selected_order,
      profileFeedback: parsedResult.improvement_tips // Now returning the single string
    };
    // --- END OF UPDATED RETURN OBJECT ---

  } catch (error) {
    console.error("Error calling Gemini API for profile curation:", error);
    console.error("Prompt sent to AI:", systemPrompt);
     // Fallback needs to return the expected structure (string for feedback)
     return {
       optimalOrder: analyzedPhotos.sort((a, b) => b.score - a.score).map(p => p.index).slice(0, 6),
       profileFeedback: `❌ Error: Could not get AI-driven feedback (${error.message}). Showing photos sorted by score.` // Return error as a single string
     };
  }
}

app.post('/optimize-profile', upload.any(), async (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ error: 'No images uploaded' });
  }

  try {
    // Map each file to a promise that extracts features or throws an error
    // Add original index to each file object before mapping
    const filesWithIndex = req.files.map((file, index) => ({ ...file, originalIndex: index }));

    const featurePromises = filesWithIndex.map(async (file) => {
      try {
        const buffer = file.buffer;
        const featData = await extractFeatures(buffer);
        // On success, return the data object with buffer/filename/index
        return {
          index: file.originalIndex, // Keep track of the original index
          features: featData.features,
          assessment: featData.assessment,
          photoType: featData.photoType,
          filename: file.originalname,
          buffer: buffer // Keep buffer temporarily if needed, maybe remove later
        };
      } catch (imageError) {
        // Log the error and add filename/index context
        const filename = file.originalname || `image ${file.originalIndex + 1}`;
        console.error(`Error extracting features for ${filename}: ${imageError.message}`);
        imageError.filename = filename; // Add filename for context later
        imageError.index = file.originalIndex; // Add index for context
        // Throw the error so Promise.allSettled catches it as 'rejected'
        throw imageError;
      }
    });

    // Wait for all feature extraction promises to settle
    const settledFeatureResults = await Promise.allSettled(featurePromises);

    // Process results: calculate scores for successful ones, handle failures
    const analysisResults = settledFeatureResults.map((result, i) => {
      // Get original index, fallback if needed (shouldn't be necessary with above changes)
      const originalIndex = filesWithIndex[i]?.originalIndex ?? i;
      const filename = filesWithIndex[i]?.originalname || `image ${originalIndex + 1}`;

      if (result.status === 'fulfilled') {
        const data = result.value;
        // Calculate score synchronously after features are extracted
        const score = Math.round(compositeScore(data) * 100);
        // Don't need buffer in the final analysis result unless specifically required later
        const { buffer, ...restOfData } = data;
        return { ...restOfData, score, error: null }; // Add score and null error
      } else {
        // Failure: construct the error object using the reason
        const reason = result.reason || new Error('Unknown feature extraction error');
        console.error(`Feature extraction failed for ${filename}: ${reason.message}`);
        return {
          index: originalIndex,
          score: 0, // Default score for failed analysis
          features: {},
          assessment: `Could not analyze ${filename}.`,
          photoType: "generic",
          filename: filename,
          error: reason.message // Include error message
        };
      }
    });

    // Filter out images that failed analysis before AI selection
    const successfulResults = analysisResults.filter(r => !r.error);

    // Call the AI curator function - it now returns tips as a single string
    const { optimalOrder, profileFeedback } = await getAISelectedOrderAndFeedback(successfulResults);

    // Map the AI's optimalOrder (indices) back to the full analysis results
    const resultsMap = new Map(analysisResults.map(r => [r.index, r]));
    const orderedSelectedImages = optimalOrder
        .map(index => resultsMap.get(index))
        .filter(Boolean);

    res.json({
      version: "v1.4-AI-Tips-String", // Update version indicator
      selectedImages: orderedSelectedImages.map(img => ({
        filename: img.filename,
        score: img.score,
        features: img.features,
        assessment: img.assessment,
        photoType: img.photoType,
        originalIndex: img.index
      })),
      profileFeedback, // Pass the single string directly
      totalImagesAnalyzed: analysisResults.length,
      successfulAnalyses: successfulResults.length,
      failedAnalyses: analysisResults.filter(r => r.error).map(r => ({ filename: r.filename, error: r.error, index: r.index }))
    });

  } catch (err) {
    // Catch errors not related to individual image processing or AI selection
    console.error("Server error in /optimize-profile:", err);
    res.status(500).json({
      error: 'profile-optimization-failed',
      details: err.message
    });
  }
});