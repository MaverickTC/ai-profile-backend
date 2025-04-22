const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json({ limit: '10mb' })); // Support base64 or image data

// Fake scoring logic
function generateFakeFeedback(imageCount) {
  const scores = [];
  const feedback = [];

  for (let i = 0; i < imageCount; i++) {
    const score = Math.floor(65 + Math.random() * 25); // 65–90%
    scores.push(score);
    feedback.push([
      "✅ Good lighting",
      "❌ Avoid group photos",
      "💡 Try smiling more",
    ]);
  }

  const order = scores
    .map((s, i) => ({ i, s }))
    .sort((a, b) => b.s - a.s)
    .map(({ i }) => i);

  return { scores, feedback, order };
}

// Route
app.post('/analyze', (req, res) => {
  const { images } = req.body; // Array of base64 strings

  if (!images || !Array.isArray(images)) {
    return res.status(400).json({ error: 'Images must be an array' });
  }

  const result = generateFakeFeedback(images.length);
  res.json(result);
});

// Start
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});