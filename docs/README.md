# 🤟 ASL Sign Language — Pure Browser Edition

Real-time ASL recognition in any browser (laptop or phone camera).  
**Zero backend required.** It runs entirely client-side using TensorFlow.js and MediaPipe Hands.

---

## 🚀 How to Deploy (Free & Permanent)

Since there is no backend, you can host this on **GitHub Pages** for free.

### Step 1: Install TF.js converter
Open your terminal and activate your virtual environment, then install TensorFlow.js:
```powershell
.venv\Scripts\activate
pip install tensorflowjs
```

### Step 2: Convert your model
Run the conversion script I made for you. It will convert your `.h5` model into a web-friendly format inside the `docs/()` folder:
```powershell
python convert_model.py
```

### Step 3: Push to GitHub Pages
1. Commit all your changes (including the new `docs/` folder) and push to your GitHub repo.
2. Go to your repository on GitHub.
3. Click **Settings** ➔ **Pages** (on the left menu).
4. Under "Build and deployment" Source, select **Deploy from a branch**.
5. Select your `main` branch, and change the folder from `/ (root)` to `/docs`.
6. Click **Save**.

Wait 1-2 minutes, and your site will be live at `https://<yourusername>.github.io/<reponame>/`!

---

## 📱 Testing Locally

If you just want to test it on your laptop before uploading:
```powershell
# Open a simple local server
cd docs
python -m http.server 8000
```
Then open `http://localhost:8000` in your browser.
