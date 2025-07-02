# Three ML-Agents

[![Live demo](https://img.shields.io/badge/web-live%20demo-brightgreen?style=flat&logo=github)](https://lukehollis.github.io/three-mlagents/)  [![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](#license)

A **JavaScript / Python** re-implementation of the core [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) examples.

![threejs_ml_agents_examples](https://github.com/user-attachments/assets/ab15555e-3f72-4992-ad82-edfe5fcd06df)


The goal of this repository is to make the ideas from Unity ML-Agents easily approachable from the browser (React + Three.js) and from lightweight Python back-ends (FastAPI) to be used for future study.

---

## Motivation

*   Learn reinforcement-learning concepts with nothing more than a web browser.
*   Provide interactive visualisations of the original ML-Agents example scenes.
*   Keep parity with the official C#/Unity implementation so that lessons learned here transfer directly to the original toolkit.

---

## Examples

### 0. Basic 1-D Move to Goal

![three_ml_agents_basic_example](https://github.com/user-attachments/assets/4fa2da3a-f983-41e0-8a66-620b2d809674)

[Live demo](https://lukehollis.github.io/three-mlagents/basic)



### 1. 3DBall Balance

![three_mlagents_3dball_example](https://github.com/user-attachments/assets/3b15e67f-daae-467a-80d3-ecdae09decd8)

[Live demo](https://lukehollis.github.io/three-mlagents/ball3d)



### 2. GridWorld Navigation

![three_ml_agents_gridworld_example](https://github.com/user-attachments/assets/eef5ae25-5189-41b1-8143-045e1d701533)

[Live demo](https://lukehollis.github.io/three-mlagents/gridworld)



### 3. Push-Block

![push_block_example](https://github.com/user-attachments/assets/825b8437-45cc-47b7-ba90-6f17ed90385c)


[Live demo](https://lukehollis.github.io/three-mlagents/push)


### 4. Wall Jump 

![wall_jump_example](https://github.com/user-attachments/assets/deb75a72-8a8a-4c94-9465-49c41c1b5f24)

[Live demo](https://lukehollis.github.io/three-mlagents/walljump)


### 5. Ant (Crawler)

![ant_example](https://github.com/user-attachments/assets/81b7ca6b-e8db-4b7e-baf1-d9d7635fa6c1)

[Live demo](https://lukehollis.github.io/three-mlagents/crawler)




## Project layout

```
three-mlagents/
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
