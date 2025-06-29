# ML-Agents – client

This folder hosts the **React + Three.js** front-end that recreates the Unity ML-Agents example scenes in the browser.

## Commands

```bash
npm install # one-time setup
npm run dev # starts Vite dev server (http://localhost:5173)
```

## Structure

* `index.html` – single-page app entry point  
* `src/main.jsx` – React renderer bootstrap  
* `src/App.jsx` – React-Router shell  
* `src/examples` – one folder per ML-Agents example

Currently implemented: **/basic** (1-D move-to-goal). 