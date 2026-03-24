# Task 2: Construction Progress Monitoring System

## Problem Statement

We need to compare two 360° photos of a construction site taken on different dates and figure out what has changed. The goal is to detect progress in specific categories like walls, floors, ceilings, etc., and make it easy for someone to review the results.

## Main Challenges

1. The photos might be taken from slightly different positions
2. Lighting could be different between the two dates
3. We need to tell the difference between actual construction progress vs just camera/lighting changes
4. Need to make the review process efficient - can't manually check everything

---

## Approach 1: Traditional Computer Vision

### Basic Pipeline

```
Photo 1 (old) + Photo 2 (new)
    ↓
Step 1: Align the images (make sure they match up)
    ↓
Step 2: Identify what's in each image (walls, floors, etc.)
    ↓
Step 3: Compare and find differences
    ↓
Step 4: Show results + flag uncertain areas for human review
```

### Step 1: Image Alignment

**Problem**: The camera might not be in exactly the same spot

**Solution**: Use feature matching to align the images
- Find common points in both images (corners, edges, etc.)
- Use ORB or SIFT algorithm to detect these features
- Calculate how to rotate/shift one image to match the other
- Apply the transformation

This is similar to how your phone stitches panorama photos together.

**Simple check**: If we can't find enough matching points (say, less than 100), flag it for manual review.

### Step 2: Identify Objects/Categories

**Problem**: Need to know what's a wall, what's a floor, etc.

**Solution**: Use a pre-trained deep learning model for semantic segmentation
- Models like Mask2Former or SegFormer work well
- These models can label each pixel in the image
- Pre-trained on general images, then fine-tune on construction photos

**Categories we care about**:
- Walls
- Floors
- Ceilings
- Doors/Windows
- Electrical fixtures
- Plumbing
- HVAC
- Structural elements (beams, columns)

**Training**: Would need ~500-1000 labeled construction images to train properly. Can start with a general model and improve over time.

**Output**: Each pixel gets a label (e.g., "wall", "floor") plus a confidence score (how sure the model is)

### Step 3: Find Changes

**Basic idea**: Compare the labeled images pixel by pixel

For each category (walls, floors, etc.):
1. Count pixels in photo 1
2. Count pixels in photo 2
3. Find the difference

**Only count high-confidence pixels** (confidence > 0.7) to avoid false positives

**Calculate progress**:
- If photo 2 has more wall pixels than photo 1 → walls have progressed
- Simple formula: `(area_new - area_old) / area_old * 100`

**Example**:
- Date 1: 10,000 wall pixels
- Date 2: 12,000 wall pixels
- Progress: (12000-10000)/10000 = 20% increase in walls

### Dealing with Noise

Big problem: Not all detected changes are real construction progress. Could be:
- Different lighting
- Camera moved slightly
- Temporary equipment in one photo
- Model mistakes

**Ways to reduce false positives**:

1. **Lighting normalization** - Adjust brightness before comparing
2. **Confidence filtering** - Only trust predictions where model is >70% confident
3. **Size filtering** - Ignore very small changes (< 100 pixels)
4. **Temporal checking** - If multiple dates available, only flag changes that appear consistently

### Manual Review Process

Can't automate everything - need humans to verify uncertain changes.

**Smart Prioritization**: Show reviewers the most important stuff first

**High priority** (check first):
- Low confidence changes (model unsure)
- Large area changes (high impact)
- Important categories (structural elements > cosmetic finishes)

**What to show reviewers**:
- Photo from Date 1
- Photo from Date 2  
- Highlighted differences
- Color coding: Green = high confidence, Yellow = medium, Red = needs review

**Goal**: Reduce review workload by 70% - only show uncertain/important cases

---

## Approach 2: Vision Language Models (Simpler & Faster)

### Why VLMs Could Work Better

Instead of training custom segmentation models, use pre-trained Vision Language Models like:
- GPT-4 Vision
- Claude 3 with vision
- Gemini Pro Vision
- LLaVA (open source)

**Advantages**:
- No training data needed
- Already understand construction scenes
- Can describe changes in natural language
- Easier to implement and iterate

### VLM-Based Pipeline

```
Photo 1 (old) + Photo 2 (new)
    ↓
Send both to VLM with prompt
    ↓
VLM analyzes and describes changes
    ↓
Parse response into structured data
    ↓
Human review
```

### Prompt Engineering Approach

**Basic Prompt Example**:
```
You are analyzing construction progress. Compare these two 360° photos 
taken on different dates.

For each category, identify what has changed:
- Walls
- Floors
- Ceilings
- Doors/Windows
- Electrical fixtures
- Plumbing
- HVAC systems
- Structural elements

Return your answer in this JSON format:
{
  "walls": {
    "status": "increased/decreased/unchanged", 
    "confidence": "high/medium/low", 
    "description": "..."
  },
  "floors": {...},
  ...
}

Be specific about what you see. If unsure, mark confidence as "low".
```

**More Detailed Prompt**:
```
Task: Compare two construction site photos and identify progress.

Step 1: Describe what you see in Photo 1 (older):
- List all visible construction elements
- Note the completion state of each

Step 2: Describe what you see in Photo 2 (newer):
- List all visible construction elements
- Note any differences from Photo 1

Step 3: Identify changes for each category:
Categories: walls, floors, ceilings, doors, windows, fixtures

For each category, state:
- What changed (specific details)
- Estimated progress (percentage if possible)
- Confidence level (high/medium/low)
- Any concerns or uncertainties

Step 4: Format as JSON.
```

### Example VLM Response

```json
{
  "walls": {
    "status": "progress",
    "confidence": "high",
    "description": "Drywall installed on north and east walls. Approximately 40% more coverage than previous photo.",
    "areas_changed": ["north wall", "east wall"]
  },
  "floors": {
    "status": "progress", 
    "confidence": "medium",
    "description": "Concrete poured in main area. Some sections still show subfloor.",
    "areas_changed": ["main floor area"]
  },
  "electrical": {
    "status": "progress",
    "confidence": "high",
    "description": "New outlet boxes installed. Count increased from 3 to 8 visible outlets.",
    "areas_changed": ["south wall"]
  },
  "notes": "Lighting significantly different between photos. Some areas harder to assess.",
  "needs_review": ["floors - partial visibility", "ceiling - lighting inconsistency"]
}
```

### Comparison: Traditional CV vs VLMs

| Aspect | Traditional CV | VLMs |
|--------|---------------|------|
| **Setup** | Complex, need training data | Simple, just API calls |
| **Accuracy** | Very precise (pixel-level) | Good (high-level) |
| **Speed** | Fast (once trained) | Few seconds per pair |
| **Cost** | High upfront, low per image | Low upfront, ~$0.01-0.02 per comparison |
| **Training Data** | 500-1000 labeled images | None needed |
| **Customization** | Full control | Limited to prompts |
| **Offline** | Yes | No (needs internet) |
| **Explanations** | None | Natural language |

---

## Which Approach to Choose?

### Use VLMs if:
- Need quick prototype
- Don't have training data
- Want natural language descriptions
- Small to medium scale
- Budget for API costs

### Use Traditional CV if:
- Need precise pixel-level measurements
- Very large scale
- Want to avoid API dependencies
- Have resources for model training
- Need offline capability

### Hybrid Approach (Best of Both):

Use VLMs for initial analysis, then traditional CV for precise measurements:

1. VLM identifies which categories changed (high-level)
2. Traditional CV measures exact areas that changed
3. Combine results for best accuracy

This gives you fast understanding AND precise metrics at lower cost.

---

## Implementation Steps

### Step 1: Proof of Concept
- Get basic alignment working
- Either: Train initial segmentation model OR set up VLM API
- Simple comparison logic
- Basic visualization
- Test on a few image pairs

### Step 2: Improve Accuracy
- For CV: Collect more training data, fine-tune model
- For VLM: Refine prompts based on results
- Add noise filtering
- Test on real sites
- Iterate based on results

### Step 3: Build Review Interface
- Simple web interface to view results
- Side-by-side comparison
- Approve/reject buttons
- Priority queue

### Step 4: Scale Up
- Move to cloud if needed
- Handle multiple sites
- Batch processing
- Monitor performance

---



## Main Challenges to Watch For

1. **Training data** (CV approach) - Need good labeled examples
2. **Prompt engineering** (VLM approach) - Getting the right format
3. **Lighting differences** - Hardest problem to solve for both
4. **Small objects** - Harder to detect accurately
5. **Camera positioning** - If too different, alignment fails

---

## Summary

This is a computer vision problem that breaks down into:
1. Align the two images
2. Identify what's in each image
3. Compare to find differences
4. Filter out noise
5. Show results for human review

**Two main approaches**:
- **Traditional CV**: More accurate but complex, needs training data
- **VLMs**: Easier to start, no training needed, good for most cases

**Recommendation**: Start with VLMs for quick results. Switch to traditional CV if you need more precision or scale.

Starting simple and iterating based on real results is better than trying to be perfect from day 1.
