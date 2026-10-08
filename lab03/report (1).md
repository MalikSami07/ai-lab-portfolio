# Lab 03 Report — Transforming Textual and Image Data into a Machine-Understandable Format

**Course:** Introduction to Artificial Intelligence
**Environment:** Python (conda env `ai-lab`), scikit-learn, Pillow, NumPy, pandas, matplotlib

---

## Part A — Bag-of-Words for text

A 3-document corpus ("The sun is shining today.", "The weather is good, the sun is great.",
"A sunny day is a wonderful day.") was converted into a document–term matrix with `CountVectorizer`.

- **Vocabulary:** 11 unique terms (`day, good, great, is, shining, sun, sunny, the, today, weather, wonderful`).
- **Matrix shape:** (3 documents × 11 terms).
- **Silent defaults checked.** The default token pattern `(?u)\b\w\w+\b` requires at least 2 word characters, so the single-letter word "a" is dropped from the vocabulary, while "the" (3 letters) is kept. `lowercase=True` by default, so case is not distinguished.
- **Sparsity:** 18 of the 33 matrix cells are zero (54.5%). With a realistic corpus of thousands of documents this figure exceeds 99%, which is why scikit-learn returns a **sparse matrix** (`scipy.sparse.csr_matrix`) rather than a dense array.

### Discussion answers (Part A)

1. Each **column** represents one unique word (term) from the vocabulary built across the whole corpus. The value in a cell is how many times that word appears in that particular document.

2. The machine-understandable format here is the **document-term matrix**: a fixed-size grid of integers, one row per document and one column per vocabulary word, where each number is a word count. This is numeric, has a consistent shape across documents, and can be fed directly into a model.

3. This is an oversimplification because it discards **word order and grammar** (Bag-of-Words cannot distinguish "sun is great" from "great is sun"), **semantics** (it treats "good" and "great" as entirely unrelated features, even though they mean something similar), and **context** (a word means the same thing to the matrix regardless of how it's used). Only the raw frequency of isolated words survives the transformation.

### Home-Lab Exercise 1 — n-grams

With `CountVectorizer(ngram_range=(1,2))` the vocabulary grew from **11** unigram features to **24** features (13 new bigrams, e.g. "day is", "good the", "is great", "shining today", "sun is", "sunny day", "the sun", "the weather", "wonderful day"). The vocabulary grows because every pair of consecutive words is now counted as its own feature alongside the individual words. This captures some local word order that plain unigrams lose, at the cost of a larger, sparser matrix, since most adjacent word pairs occur in only one document.

### Home-Lab Exercise 2 — raw counts favor long/repetitive documents

A fourth document, "sun sun sun sun sun", was added. Its count for the feature "sun" is **5**, versus 1 or 0 for the other three documents. Because `CountVectorizer` uses raw word counts, a document's score for a word scales directly with how many times that word is repeated, independent of whether the word is actually central to the document's meaning. A long or repetitive document therefore dominates that feature purely through repetition. This is why raw counts favor long documents, and it motivates normalizing by document length (term frequency) or using TF–IDF instead of raw counts.

---

## Part B — Pixel features for images

An image (148×148 RGB, loaded from file; a 100×100 RGB random image is used as a fallback when no file is found) was converted through: **RGB array → grayscale → flattened vector**.

| Stage | Shape | Feature count |
|---|---|---|
| Original RGB | (148, 148, 3) | 65,712 |
| Grayscale | (148, 148) | 21,904 |
| Flattened vector | (21,904,) | 21,904 |

Grayscale conversion is a **3-to-1 feature reduction**: it removes two-thirds of the raw pixel values (the color channels) while keeping spatial/intensity structure.

### Discussion answers (Part B)

1. Grayscale conversion is feature engineering because it deliberately reduces the input from three numbers per pixel (R, G, B) to one (luminance), cutting the feature count by a factor of three while keeping the information most models need — shape and intensity — intact. It trades away color to simplify the input and reduce computation.

2. The length of the final flattened vector is **H × W** — one value per pixel, since grayscale leaves a single channel.

3. Flattening is necessary because models like Logistic Regression and SVM expect a single **one-dimensional feature vector** per sample, not a 2D grid. Flattening converts the H×W matrix into a length-(H×W) vector in a fixed, reproducible order, which is the format these algorithms accept.

4. Resizing every image to a consistent size (e.g. 64×64) before grayscale and flattening guarantees that **every sample produces a feature vector of the same length**, regardless of the original photo's dimensions. Without this, images of different sizes would flatten into vectors of different lengths, which no standard model can handle, since every sample must have the same number of features.

### Home-Lab Exercise 3 — resizing a real photograph

A real photograph (816×1276) was resized to 64×64 and 32×32 and compared visually.

| Size | Flattened vector length |
|---|---|
| Original (816×1276) | 1,041,216 |
| 64×64 | 4,096 |
| 32×32 | 1,024 |

At 64×64 the photo is still clearly recognizable — facial features and the card's layout are legible. At 32×32 the image becomes visibly blocky: fine detail is gone, text on the card is no longer readable, and the face is reduced to rough blobs of shading rather than distinct features. Smaller fixed sizes make the feature vector shorter and cheaper to process, but destroy exactly the fine-grained spatial detail a model would need to tell similar images apart.

### Home-Lab Exercise 4 — `image_to_features` consistency check

```python
def image_to_features(path, size=(64, 64)):
    """Load an image, resize to a fixed shape, convert to grayscale, and flatten."""
    img = Image.open(path).convert("RGB").resize(size)
    gray = img.convert("L")
    return np.array(gray).flatten()
```

Three differently sized versions of the same photo (200×300, 816×1276, and 50×50) were each passed through `image_to_features`. All three returned a vector of length **4,096**, confirming the function guarantees a fixed output length regardless of the input image's original dimensions — exactly what is needed to build a consistent feature matrix across a dataset of differently sized images.

---

## Home Assignment — 20-image labelled dataset

Two classes (10 images each) were collected and passed through `image_to_features` at `size=(64, 64)`.

- **Dataset shape:** `X.shape = (20, 4096)`, `y.shape = (20,)`
- **Samples per feature:** 20 / 4096 ≈ **0.0049**
- Saved to `dataset.npz` (`X`, `y`).

### Paragraph — is 20 samples in 4096 dimensions a sound basis for learning?

Twenty samples in a 4096-dimensional feature space is not a sound basis for learning. With far more dimensions than samples (about 0.005 samples per feature), the data is extremely sparse relative to the space it occupies: almost any two points can look "close" or "far" depending on which pixels happen to differ, and a model has vastly more freedom to fit noise than genuine structure. This is the **curse of dimensionality**: in such a high-dimensional space, distance-based methods lose meaning, and flexible models will memorize the 20 training examples rather than learn a generalizable rule, since there are effectively infinite ways to separate 20 points in 4096 dimensions. A model trained this way would likely show excellent training accuracy and poor performance on any new image. Laboratory 6's dimensionality reduction (e.g., PCA) would address this directly by projecting the 4096 raw pixel features down to a much smaller number of components that capture most of the meaningful variation, shrinking the feature space toward something the sample size can actually support. A CNN (Laboratory 8) takes a different approach: rather than flattening pixels into an unordered vector and hoping a simple model finds structure in it, convolutional layers exploit the image's spatial structure directly, learning local patterns (edges, textures) with far fewer independent parameters relative to the raw pixel count, and are typically trained on thousands of images rather than twenty. Either approach is a more appropriate response to this dataset's shape than training directly on raw flattened pixels.

---

## Summary of all seven discussion questions

| # | Part | Question | Answered |
|---|---|---|---|
| 1 | A | What does each column represent? | ✅ |
| 2 | A | What is the machine-understandable format? | ✅ |
| 3 | A | How is this an oversimplification? | ✅ |
| 4 | B | How is grayscale conversion feature engineering? | ✅ |
| 5 | B | Length of the flattened vector for H×W? | ✅ |
| 6 | B | Why is flattening necessary? | ✅ |
| 7 | B | Advantage of resizing to a consistent size? | ✅ |

## Deliverables checklist

- [x] `lab03/Lab_03_Text_and_Image_Features.ipynb` — Parts A & B executed
- [x] `image_to_features` function defined and verified (Exercise 4)
- [x] `lab03/dataset.npz` — 20-image labelled dataset (X: 20×4096, y: 20)
- [x] `lab03/report.md` — this file, all seven discussion questions + exercises
