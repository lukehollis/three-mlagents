
![Screenshot 2025-06-29 at 10 25 21](https://github.com/user-attachments/assets/501626cc-fec4-4a0d-8d6c-6f50ff320575)

# Three ML-Agents

A **JavaScript / Python** re-implementation of the core [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) examples.

The goal of this repository is to make the ideas from Unity ML-Agents easily approachable from the browser (React + Three.js) and from lightweight Python back-ends (FastAPI) to be used for future study.

---

## Motivation

*   Learn reinforcement-learning concepts with nothing more than a web browser.
*   Provide interactive visualisations of the original ML-Agents example scenes.
*   Keep parity with the official C#/Unity implementation so that lessons learned here transfer directly to the original toolkit.

---

## Project layout

```
rlagents/
│
├─ client/         ← Vite + React + Three.js front-end (examples live here)
│   └─ src/examples
│       ├─ Index.jsx  ← landing page listing all examples
│       └─ Basic.jsx  ← 1-D "move-to-goal" environment matching Unity Basic
│
├─ api/            ← FastAPI micro-service (Python ≥3.9)
│   ├─ main.py      ← gym-style REST API for each environment
│   └─ requirements.txt
│
└─ ml-agents/      ← Upstream Unity project kept for reference only (no build step)
```

You only need **client** and **api** to run the demos. The `ml-agents` directory remains untouched so you can cross-reference the original C#/Unity code (see `Examples/Basic/Scripts/BasicController.cs` for this particular demo).

---

## Quick start

### 1. Front-end (browser)

```bash
cd client
npm install        # installs React, Three.js, @react-three/fiber, @react-three/drei …
npm run dev        # opens http://localhost:5173
```

You will land on `/` – a list of examples. Click **/basic** to open the first scene. Use **← / →** to move the agent cube; rewards replicate the logic of the original `BasicController` script:

* Each step: **−0.01**
* Small goal (position 7): **+0.1** and episode terminates
* Large goal (position 17): **+1** and episode terminates

### 2. Back-end (optional)

For browser-only play the API is **not required**. If you want to let external RL algorithms interact with the environment, spin up the FastAPI service:

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

Endpoints:

* `POST /basic/reset` → `{ "position": 10 }`
* `POST /basic/step`  → `{ "position": int, "reward": float, "done": bool }`

The contract mirrors OpenAI Gym (`position` plays the role of the observation).

---

## Roadmap

1. **/basic** 1-D movement – _done_
2. **/3dball** continuous control & physics
3. Curriculum learning (+ curriculum visualiser)
4. On-policy & off-policy training notebook examples

---

## License

MIT © 2025
