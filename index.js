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

async function extractFeatures(buffer, profileContext = {}) {
  const resizedBuffer = await sharp(buffer)
    .resize({ width: 512 })
    .png({ quality: 80 })
    .toBuffer();

  const base64Image = resizedBuffer.toString('base64');

  let contextText = "";
  if (Object.keys(profileContext).length > 0) {
    contextText = "User Profile Context (use this to better understand the user's intent and target audience for this photo):\n";
    if (profileContext.goal) contextText += `- Goal: ${profileContext.goal}\n`;
    if (profileContext.gender) contextText += `- Gender: ${profileContext.gender}\n`;
    if (profileContext.interestedIn) contextText += `- Interested In: ${profileContext.interestedIn}\n`;
    if (profileContext.appsUsed && profileContext.appsUsed.length > 0) contextText += `- Apps Used: ${profileContext.appsUsed.join(', ')}\n`;
    if (profileContext.ageRange) contextText += `- Preferred Age Range: ${profileContext.ageRange}\n`;
    if (profileContext.bio) contextText += `- Bio: "${profileContext.bio}"\n`;
    if (profileContext.prompts && profileContext.prompts.length > 0) {
        contextText += `- Prompts:\n${profileContext.prompts.map(p => `  - "${p}"`).join('\n')}\n`;
    }
    contextText += "\n";
  }

  const systemPrompt = `${contextText}You are a dating profile photo evaluator with expertise in what makes an ideal dating profile image. Extract the following features from the image:

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
  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

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
  const defaultFeatures = {
    quality: 0,
    aesthetics: 0,
    smileProb: 60, // Neutral if not visible/undefined, as per original prompt
    gazeDeg: 90,   // Max degrees, least favorable for gazeScore if undefined
    redFlag: false,
    petFlag: false,
    filterStrength: 0, // No filter if undefined
    numFaces: 1,       // Assume 1 face if undefined
    postureScore: 0    // Neutral posture if undefined
  };
  // Merge provided features with defaults. data.features might be undefined or an empty object.
  const f = { ...defaultFeatures, ...(data.features || {}) };
  const type = data.photoType || "generic"; // Default to generic if photoType is missing
  
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
    W.postureBonus * f.postureScore // f.postureScore will use default 0 if not present
  );

  const penalties = (
    W.redFlag * (f.redFlag ? 1 : 0) +
    W.filterPenalty * f.filterStrength + // f.filterStrength will use default 0
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

  const systemPrompt = `You are a dating photo coach. Your feedback should be extremely concise and actionable, formatted as JSON.
The photo has a score of ${score}/100.
AI's assessment of the photo:
${assessment}

CRITICAL RULES:
- NEVER suggest adding different photos or replacing this photo.
- If face is not visible, DO NOT suggest showing face in another photo.
- ONLY comment on THIS SPECIFIC PHOTO.
- If the photo has limitations (like face not visible) but works well for its type (e.g., showing a hobby), focus on its positive aspects for that type.

RESPONSE FORMAT:
Return ONLY a valid JSON object with two keys: "good_points" and "improvement_points".
- "good_points": An array of 3-5 strings. Each string is a strength of the photo, starting with a "👍" emoji.
- "improvement_points": An array of 2-5 strings. Each string is a specific, actionable improvement, starting with a "👎" emoji.
- Each point (both good and improvement) MUST be a very short phrase (2-6 words).
- Examples for points: "👍 Great smile", "👍 Shows personality", "👎 Try different angle", "👎 Too dark", "👎 Blurry background".
- DO NOT use full sentences. Be direct.
- If there are no clear improvements, provide at least 1-2 minor suggestions.
- For strong photos (score > 75), provide at least 3 good points.
- For weaker photos (score < 50), provide at least 3 improvement points.

Example JSON Output:
\`\`\`json
{
  "good_points": ["👍 Active setting", "👍 Clear action shot", "👍 Sporty vibe", "👍 Good posture"],
  "improvement_points": ["👎 Enhance brightness", "👎 Center the frame", "👎 Avoid cluttered background", "👎 Use more vibrant colors"]
}
\`\`\`
Ensure your entire response is ONLY the JSON object.`;

  try {
    // Initialize Gemini model
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

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
          imagePart
        ]
      }],
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 400, // Increased for more points
        responseMimeType: "application/json", // Expect JSON response
      }
    });

    const response = result.response;
    const responseText = response.text();
    let parsedResult;

    try {
      parsedResult = JSON.parse(responseText);
    } catch (e) {
      console.error("Failed to parse JSON feedback from Gemini:", responseText, e);
      throw new Error("Invalid JSON feedback from AI.");
    }

    if (!parsedResult || typeof parsedResult.good_points === 'undefined' || typeof parsedResult.improvement_points === 'undefined') {
        console.error("AI response missing required keys (good_points, improvement_points):", parsedResult);
        throw new Error("AI response missing required keys.");
    }
    
    // Instead of combining the arrays, return them separately
    return {
      good_points: Array.isArray(parsedResult.good_points) ? parsedResult.good_points : ["👍 Great photo!"],
      improvement_points: Array.isArray(parsedResult.improvement_points) ? parsedResult.improvement_points : []
    };
  } catch (error) {
    console.error("Error calling Gemini API for feedback generation:", error);
    throw error;
  }
}

const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }
});

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json());

app.post('/analyze', (req, res) => {
  upload.any()(req, res, async function(err) {
    if (err) {
      if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({ 
          error: 'file-too-large',
          details: `File size exceeds the ${Math.floor(upload.limits.fileSize / (1024 * 1024))}MB limit.`
        });
      }
      return res.status(500).json({ error: 'upload-failed', details: err.message });
    }
    
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No images uploaded' });
    }

    let profileContext = {};
    if (req.body.profile_context_json) {
      try {
        profileContext = JSON.parse(req.body.profile_context_json);
        console.log("Received profile context for /analyze:", profileContext);
      } catch (e) {
        console.warn("Could not parse profile_context_json for /analyze:", e.message);
      }
    }

    try {
      // Map each file to a promise that resolves with its analysis result or throws an error
      const analysisPromises = req.files.map(async (file) => {
        try {
          const buffer = file.buffer;
          const featData = await extractFeatures(buffer, profileContext);
          const score = Math.round(compositeScore(featData) * 100);
          const feedback = await generateFeedback(featData, score, buffer);

          // On success, return the data object directly
          return {
            score,
            feedback, // This is now an object with good_points and improvement_points
            features: featData.features,
            assessment: featData.assessment,
            photoType: featData.photoType
          };
        } catch (imageError) {
          console.error(`Error processing image ${file.originalname}: ${imageError.message}`);
          imageError.filename = file.originalname;
          throw imageError;
        }
      });

      const settledResults = await Promise.allSettled(analysisPromises);

      // Process the results from Promise.allSettled
      const results = settledResults.map((result, index) => {
        if (result.status === 'fulfilled') {
          return result.value;
        } else {
          const reason = result.reason || new Error('Unknown analysis error');
          const filename = reason.filename || req.files?.[index]?.originalname || `image ${index + 1}`;
          console.error(`Analysis failed for ${filename}: ${reason.message}`);
          return {
            score: null,
            feedback: {
              good_points: [],
              improvement_points: [`👎 Error processing ${filename}: ${reason.message}`]
            },
            features: null,
            assessment: `Could not analyze ${filename}.`,
            photoType: "generic",
            error: reason.message
          };
        }
      });

      // Calculate order based on scores
      const order = results
        .map((r, i) => ({ i, score: r.score ?? -1 }))
        .sort((a, b) => b.score - a.score)
        .map(o => o.i);

      // Extract data arrays for the response
      const scores = results.map(r => r.score);
      const goodFeedback = results.map(r => r.feedback.good_points || []);
      const improvementFeedback = results.map(r => r.feedback.improvement_points || []);
      const features = results.map(r => r.features);
      const assessments = results.map(r => r.assessment);
      const photoTypes = results.map(r => r.photoType);

      // Calculate overall profile score
      const profileScore = calculateOverallProfileScore(scores, photoTypes);

      res.json({
        version: "v1.2",
        scores,
        feedback: {
          good_points: goodFeedback,
          improvement_points: improvementFeedback
        },
        order,
        features,
        assessments,
        photoTypes,
        profileScore
      });

    } catch (err) {
      console.error("Server error in /analyze:", err);
      res.status(500).json({
        error: 'analysis-failed',
        details: err.message
      });
    }
  });
});

// NEW FUNCTION: Uses AI to select and order photos based on analysis data
async function getAISelectedOrderAndFeedback(photosForSelection, profileContext = {}) {
  if (!photosForSelection || photosForSelection.length === 0) {
    return {
      optimalOrder: [],
      improvementSteps: [],
      suggestedPrompts: []
    };
  }

  let contextText = "";
  if (Object.keys(profileContext).length > 0) {
    contextText = "User Profile Context:\n";
    if (profileContext.goal) contextText += `- Goal: ${profileContext.goal}\n`;
    if (profileContext.gender) contextText += `- Gender: ${profileContext.gender}\n`;
    if (profileContext.interestedIn) contextText += `- Interested In: ${profileContext.interestedIn}\n`;
    if (profileContext.appsUsed && profileContext.appsUsed.length > 0) contextText += `- Apps Used: ${profileContext.appsUsed.join(', ')}\n`;
    if (profileContext.ageRange) contextText += `- Preferred Age Range: ${profileContext.ageRange}\n`;
    //if (profileContext.bio) contextText += `- Bio: "${profileContext.bio}"\n`;
    //if (profileContext.prompts && profileContext.prompts.length > 0) {
        //contextText += `- Prompts:\n${profileContext.prompts.map(p => `  - "${p}"`).join('\n')}\n`;
    }
    contextText += "\n";

  const systemPrompt = `${contextText}You are an expert dating profile photo curator. Your task is to select and order the optimal set of photos for a dating profile.

SELECTION CRITERIA (in order of importance):
1. QUALITY: Select photos that are clear, well-lit, and high resolution, and show the person's face in a good handsome/beatiful way
2. FACE VISIBILITY: At least one photo should clearly show the person's face
3. DIVERSITY: Include a mix of photo types following this priority order: ${SLOT_ORDER.join(' > ')}
4. CONSISTENCY: All photos should appear to be of the same person and from a similar time period
5. AUTHENTICITY: Photos should look natural and represent the person accurately

REJECTION CRITERIA:
- Blurry or poorly lit images
- Photos where the subject is too small or distant
- Heavily filtered or edited photos
- Multiple similar photos (select the best one)
- Photos with unflattering expressions or awkward poses

ORDERING GUIDELINES:
1. First photo: Clear headshot with good lighting and friendly expression
2. Second photo: Full body shot showing physique and style
3. Third photo: Activity/hobby photo showing interests
4. Fourth photo: Social photo showing friendliness
5. Fifth photo: Another high-quality photo showing a different aspect
6. Sixth photo: Any remaining strong photo that adds variety

Note: If any specific photo type is missing, substitute with the next best available photo that shows a different aspect of the person. (It is mostly better to have more photos unless they are really bad)

IMPROVEMENT RECOMMENDATIONS:
Based on the selected photos, identify 3-5 specific ways the profile could be improved.
- Be specific and actionable
- Check carefully what's already in the photos before making recommendations
- Focus on quality improvements or diversity rather than suggesting duplicates

PROMPT SUGGESTIONS, choose from these options and create personalized answers:
- "I go crazy for"
- "A life goal of mine"
- "My simple pleasures"
- "Green flags I look for"
- "Try to guess this about me"
- "I wind down by"
- "Together, we could"
- "Most spontaneous thing I've done"
- "I geek out on"
- "I'll brag about you to my friends if"

RESPONSE FORMAT:
- Select a maximum of 6 photos. If fewer than 6 are suitable or available, select those.
- Provide your response ONLY as a JSON object with three keys:
  - "selected_order": An array of the original photo indices (e.g., [3, 0, 5, 1, 4, 2]) representing your chosen photos in the optimal display order.
  - "improvement_steps": An array of 3-5 specific improvement recommendations by looking at the selected photos, each an object with:
      - "title": A short, clear title (e.g., "Photo with Friends", "Candid Photo")
      - "description": A brief explanation (e.g., "Add a picture with a friend to your profile.")
  - "suggested_prompts": An array of 3 objects, each with:
      - "prompt": One of the prompt options listed above
      - "answer": A personalized, authentic-sounding answer (1-3 sentences) based on what you can infer about the person from their photos

Focus on making objective selections based on technical quality and dating profile best practices.`;

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    const userParts = [];
    userParts.push({ text: systemPrompt });

    for (const photo of photosForSelection) {
      if (!photo.buffer) {
        console.warn(`Skipping photo index ${photo.index} for AI selection due to missing buffer.`);
        userParts.push({ text: `Photo Original Index: ${photo.index} - Image data was not available for review.\n---` });
        continue;
      }
      try {
        const resizedBuffer = await sharp(photo.buffer)
          .resize({ width: 512, withoutEnlargement: true })
          .png({ quality: 80 })
          .toBuffer();
        const base64Image = resizedBuffer.toString('base64');

        userParts.push({ text: `Photo Original Index: ${photo.index}\n---` });
        userParts.push({
          inlineData: {
            data: base64Image,
            mimeType: "image/png"
          }
        });
      } catch (imgError) {
        console.error(`Error processing image index ${photo.index} for AI selection: ${imgError.message}. Skipping this image for AI review.`);
        userParts.push({ text: `Photo Original Index: ${photo.index} - Could not be processed for AI review due to an error: ${imgError.message}\n---` });
      }
    }
    
    userParts.push({ text: "Please provide your selection, improvement tips, and suggested prompts based on the images shown above." });

    const result = await model.generateContent({
      contents: [{
        role: "user",
        parts: userParts
      }],
      generationConfig: {
        temperature: 0.6,
        maxOutputTokens: 1000,
        responseMimeType: "application/json",
      }
    });

    const response = result.response;
    const responseText = response.text();
    const parsedResult = JSON.parse(responseText);

    if (!parsedResult.selected_order || !Array.isArray(parsedResult.improvement_steps)) {
        throw new Error("AI response missing required keys or has invalid format.");
    }
    
    if (!parsedResult.selected_order.every(n => !isNaN(n) && typeof n === 'number')) {
        throw new Error("AI response 'selected_order' contains non-numeric values.");
    }

    // Validate improvement_steps structure
    if (!parsedResult.improvement_steps.every(step => 
      typeof step === 'object' && 
      typeof step.title === 'string' && 
      typeof step.description === 'string')) {
      throw new Error("AI response 'improvement_steps' has invalid format.");
    }

    // Validate suggested_prompts structure if present
    const suggestedPrompts = parsedResult.suggested_prompts || [];
    if (suggestedPrompts.length > 0 && !suggestedPrompts.every(prompt => 
      typeof prompt === 'object' && 
      typeof prompt.prompt === 'string' && 
      typeof prompt.answer === 'string')) {
      console.warn("AI response 'suggested_prompts' has invalid format, using empty array instead.");
      parsedResult.suggested_prompts = [];
    }

    return {
      optimalOrder: parsedResult.selected_order,
      improvementSteps: parsedResult.improvement_steps,
      suggestedPrompts: parsedResult.suggested_prompts || []
    };

  } catch (error) {
    console.error("Error calling Gemini API for profile curation:", error);
    console.error("System Prompt sent to AI:", systemPrompt);
    return {
      optimalOrder: photosForSelection.map(p => p.index).slice(0, 6),
      improvementSteps: [],
      suggestedPrompts: []
    };
  }
}

app.post('/optimize-profile', (req, res) => {
  console.log("Received request to /optimize-profile");
  upload.any()(req, res, async function(err) {
    if (err) {
      console.error(`Upload error: ${err.message}`);
      if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({ 
          error: 'file-too-large',
          details: `File size exceeds the ${Math.floor(upload.limits.fileSize / (1024 * 1024))}MB limit.`
        });
      }
      return res.status(500).json({ error: 'upload-failed', details: err.message });
    }
    
    if (!req.files || req.files.length === 0) {
      console.warn("No images uploaded to /optimize-profile");
      return res.status(400).json({ error: 'No images uploaded' });
    }

    console.log(`Processing ${req.files.length} images for profile optimization`);
    
    let profileContext = {};
    if (req.body.profile_context_json) {
      try {
        profileContext = JSON.parse(req.body.profile_context_json);
        console.log("Received profile context for /optimize-profile:", profileContext);
      } catch (e) {
        console.warn("Could not parse profile_context_json for /optimize-profile:", e.message);
      }
    }

    try {
      // Skip detailed analysis and just prepare the images for AI selection
      const imagesForSelection = req.files.map((file, index) => ({
        index: index,
        buffer: file.buffer,
        filename: file.originalname
      }));
      
      const { optimalOrder, improvementSteps, suggestedPrompts } = await getAISelectedOrderAndFeedback(imagesForSelection, profileContext);

      // Ensure we have at least one image in the optimal order
      let finalOrder = optimalOrder;
      if (!finalOrder || finalOrder.length === 0) {
        console.warn("AI returned empty selection, defaulting to first image");
        // Default to the first image if AI selection is empty
        finalOrder = [0];
      }
      
      console.log(`Final order after validation: ${JSON.stringify(finalOrder)}`);

      // Calculate a profile score for the optimized selection
      // Since we don't have scores here, we'll use a simpler calculation
      // or you could analyze the selected images first
      const profileScore = Math.min(100, 70 + finalOrder.length * 5); // Simple placeholder calculation
      console.log(`Calculated profile score: ${profileScore}`);

      const response = {
        version: "v2.0-AI-Image-Selection-With-Prompts",
        selectedImages: finalOrder.map(index => ({
          originalIndex: index,
          filename: req.files[index]?.originalname || `image_${index}`
        })),
        improvementSteps: improvementSteps || [],
        suggestedPrompts: suggestedPrompts,
        totalImagesAnalyzed: req.files.length,
        profileScore
      };
      
      console.log(`Sending response with ${response.selectedImages.length} selected images, ${response.improvementSteps.length} improvement steps, and ${response.suggestedPrompts.length} suggested prompts`);
      res.json(response);

    } catch (err) {
      console.error("Server error in /optimize-profile:", err);
      console.error("Error stack:", err.stack);
      
      // Even in case of error, return at least the first image
      const fallbackSelection = [{
        originalIndex: 0,
        filename: req.files[0]?.originalname || "image_0"
      }];
      
      // Default prompts for error case
      const fallbackPrompts = [
        {
          "prompt": "I geek out on …",
          "answer": "Finding hidden gems in my city - whether it's a cozy bookstore, a scenic hiking trail, or a café with the perfect atmosphere."
        },
        {
          "prompt": "Together, we could …",
          "answer": "Explore new restaurants, debate the best movies of all time, and create memories worth sharing."
        },
        {
          "prompt": "My simple pleasures",
          "answer": "A good book, sunset walks, and conversations that last longer than planned."
        }
      ];
      
      console.log("Using fallback selection with first image due to error");
      res.json({
        version: "v2.0-AI-Image-Selection-With-Prompts",
        selectedImages: fallbackSelection,
        improvementSteps: [
          {
            "title": "Add more photos",
            "description": "Include at least 4-6 photos in your profile."
          },
          {
            "title": "Show your face clearly",
            "description": "Make sure your face is clearly visible in at least one photo."
          }
        ],
        suggestedPrompts: fallbackPrompts,
        totalImagesAnalyzed: req.files.length,
        profileScore: 50 // Default score for error case
      });
    }
  });
});

// Add this function to calculate overall profile score
function calculateOverallProfileScore(scores, photoTypes) {
  if (!scores || scores.length === 0) {
    return 0;
  }

  // Weight different photo types differently
  const typeWeights = {
    "primary_headshot": 1.2,
    "full_body": 1.1,
    "hobby_activity": 1.0,
    "pet": 0.9,
    "group_social": 0.9,
    "generic": 0.8
  };

  // Calculate weighted average
  let totalWeight = 0;
  let weightedSum = 0;

  for (let i = 0; i < scores.length; i++) {
    if (scores[i] === null) continue; // Skip null scores
    
    const score = scores[i];
    const type = photoTypes[i] || "generic";
    const weight = typeWeights[type] || 0.8;
    
    weightedSum += score * weight;
    totalWeight += weight;
  }

  // Calculate base score (0-100)
  const baseScore = totalWeight > 0 ? Math.round(weightedSum / totalWeight) : 0;
  
  // Apply bonuses/penalties based on profile completeness
  let finalScore = baseScore;
  
  // Bonus for having at least 4 photos
  if (scores.filter(s => s !== null).length >= 4) {
    finalScore += 5;
  }
  
  // Bonus for having diverse photo types
  const uniqueTypes = new Set(photoTypes.filter(t => t)).size;
  if (uniqueTypes >= 3) {
    finalScore += 5;
  }
  
  // Cap at 100
  return Math.min(100, finalScore);
}

app.listen(PORT, () => {
  console.log(`AI Profile Backend running on port ${PORT}`);
  console.log(`API endpoints:`);
  console.log(`- POST /analyze: Analyze individual photos`);
  console.log(`- POST /optimize-profile: Get AI-optimized profile`);
});