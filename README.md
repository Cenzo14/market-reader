# Market Reader — setup (takes ~5 minutes, $0)

Needs the Claude desktop app or a computer (not the phone).

1. Go to https://github.com/new
   - Repository name: market-reader
   - Set to Public
   - Click green "Create repository" button (bottom of page)
2. On the new repo page, click "uploading an existing file" (blue link in the middle of the page)
   - Drag in these 3 files: app.py, requirements.txt, README.md
   - Click green "Commit changes" button (bottom)
3. Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "Create app" (top-right) → "Deploy a public app from GitHub"
   - Repository: market-reader
   - Branch: main
   - Main file path: app.py
   - Click "Deploy"
4. Wait ~2 minutes. Your app link appears at the top of the screen.

Type any ticker in the left sidebar (AAPL, NVDA, BRK-B). Tabs run left to right.
