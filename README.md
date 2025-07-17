# Three ML-Agents

[![Live demo](https://img.shields.io/badge/web-live%20demo-brightgreen?style=flat&logo=github)](https://lukehollis.github.io/three-mlagents/)
![GitHub stars](https://img.shields.io/github/stars/lukehollis/three-mlagents?style=flat&logo=github)
![GitHub last commit](https://img.shields.io/github/last-commit/lukehollis/three-mlagents?style=flat&logo=github)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](#license)

![React](https://img.shields.io/badge/React-18+-61dafb?style=flat&logo=react)
![Three.js](https://img.shields.io/badge/Three.js-r150+-000000?style=flat&logo=three.js)
![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?style=flat&logo=fastapi)

A **JavaScript / Python** re-implementation of the core [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) examples.

![glider_example](https://github.com/user-attachments/assets/1c82281a-6982-4d05-95e4-816e6b8f61b4)


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

### 6. Swimmer / Worm

![worm_example](https://github.com/user-attachments/assets/6ae84aac-ef38-4e79-a2e9-b4079b61bb8e)

[Live demo](https://lukehollis.github.io/three-mlagents/worm)

### 7. Brick Break

![brick_break_example](https://github.com/user-attachments/assets/790b1bbf-3758-4e6c-b570-7dd268b6f987)


[Live demo](https://lukehollis.github.io/three-mlagents/brickbreak)

### 8. Food Collector

![food_collector_example](https://github.com/user-attachments/assets/199505ec-361e-475c-9f15-94becf525599)


[Live demo](https://lukehollis.github.io/three-mlagents/foodcollector)

### 9. Bicycle

![bicycle_example](https://github.com/user-attachments/assets/929a8c8d-8154-4f35-9af9-95b78940c9be)


[Live demo](https://lukehollis.github.io/three-mlagents/bicycle)

### 10. Glider

![glider_example](https://github.com/user-attachments/assets/1c82281a-6982-4d05-95e4-816e6b8f61b4)


[Live demo](https://lukehollis.github.io/three-mlagents/glider)


### 11. MineCraft

![minecraft_example](https://github.com/user-attachments/assets/1338e926-8c8d-412e-bc2c-0627e47df165)


[Live demo](https://lukehollis.github.io/three-mlagents/minecraft)


### 12. Fish

![fish_example](https://github.com/user-attachments/assets/df5e947f-cbe1-401a-a491-180159515acc)


[Live demo](https://lukehollis.github.io/three-mlagents/fish)


### 13. Intersection

![intersection_example](https://github.com/user-attachments/assets/0f9c6b90-2a62-466c-af83-0f695e231671)


[Live demo](https://lukehollis.github.io/three-mlagents/intersection)




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

## MineCraft RL + LLM Example - Additional Setup

The **MineCraft** example (#11) combines reinforcement learning with large language models (LLMs) for intelligent agent behavior. This requires additional setup beyond the basic installation.

### Option 1: Local Ollama (Recommended for Development)

1. **Install Ollama**: Download and install from [ollama.ai](https://ollama.ai/)

2. **Pull the required model**:
   ```bash
   ollama pull gemma3n:latest
   ```

3. **Configure the example**: In `api/examples/minecraft.py`, set:
   ```python
   USE_LOCAL_OLLAMA = True
   ```

4. **Start Ollama**: Make sure Ollama is running locally (usually starts automatically)

### Option 2: OpenRouter (Cloud-based)

1. **Get an OpenRouter API key**: Sign up at [openrouter.ai](https://openrouter.ai/) and get your API key

2. **Set environment variable**:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```

3. **Configure the example**: In `api/examples/minecraft.py`, set:
   ```python
   USE_LOCAL_OLLAMA = False
   ```

4. **Note**: This option uses `anthropic/claude-sonnet-4` and requires credits/tokens

### Additional Dependencies

The MineCraft example also requires additional Python packages for embeddings and LLM functionality. Make sure these are included in your `api/requirements.txt`:

```
sentence-transformers  # for text embeddings
openai                 # for LLM API calls (if using OpenRouter)
```

### Running the MineCraft Example

1. Start the API with LLM support:
   ```bash
   cd api
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

2. Open the frontend and navigate to the MineCraft example

3. The agents will use LLMs for strategic decision-making, communication, and trading while using RL for basic movement and mining actions.

---

## License

MIT © 2025
