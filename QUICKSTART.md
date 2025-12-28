# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Get OpenRouter API Key

1. Go to https://openrouter.ai
2. Sign up for a free account
3. Get your API key from the dashboard
4. Set it as an environment variable:

**Windows PowerShell:**
```powershell
$env:OPENROUTER_API_KEY='your-api-key-here'
```

Or with double quotes (if needed):
```powershell
$env:OPENROUTER_API_KEY="your-api-key-here"
```

**Note:** You can skip this step and enter your API key directly in the Streamlit interface sidebar instead!

**Windows CMD:**
```cmd
set OPENROUTER_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Step 3: Run the Application

### Option A: Streamlit Interface (Recommended)

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Option B: Command Line

```bash
python rag_pipeline.py
```

## Step 4: Use the Assistant

1. In the Streamlit interface, enter your API key in the sidebar
2. Click "Initialize/Reload RAG Pipeline"
3. Wait for the index to build (first time only)
4. Ask questions like:
   - "What factors affect construction project delays?"
   - "What safety protocols must be followed?"
   - "How long does a typical residential project take?"

## Troubleshooting

### "OPENROUTER_API_KEY not set"
- Make sure you've set the environment variable correctly
- Or enter it directly in the Streamlit sidebar

### "Document directory not found"
- Make sure the `documents/` folder exists with `.txt` files
- Check that you're running from the project root directory

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- If using a virtual environment, make sure it's activated

### Index build takes long time
- This is normal on first run
- The index will be saved and reused on subsequent runs

