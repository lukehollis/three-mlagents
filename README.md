# Three ML-Agents

[![Live demo](https://img.shields.io/badge/web-live%20demo-brightgreen?style=flat&logo=github)](https://lukehollis.github.io/three-mlagents/)
![GitHub stars](https://img.shields.io/github/stars/lukehollis/three-mlagents?style=flat&logo=github)
![GitHub last commit](https://img.shields.io/github/last-commit/lukehollis/three-mlagents?style=flat&logo=github)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](#license)

![React](https://img.shields.io/badge/React-18+-61dafb?style=flat&logo=react)
![Three.js](https://img.shields.io/badge/Three.js-r150+-000000?style=flat&logo=three.js)
![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?style=flat&logo=fastapi)

A **JavaScript / Python** re-implementation of the core [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) examples. Test out the [Live Demos](https://lukehollis.github.io/three-mlagents/).


![three_mlagents](https://github.com/user-attachments/assets/0589bada-85b7-4580-946a-99469e2b9b8e)



The goal of this repository is to make the ideas from Unity ML-Agents easily approachable from the browser (React + Three.js) and from lightweight Python back-ends (FastAPI) to be used for future study.



## Examples

View all the demos on the live demos page: [Live Demos](https://lukehollis.github.io/three-mlagents/). 

Select demos are here. 


### MineCraft

![minecraft_example](https://github.com/user-attachments/assets/1338e926-8c8d-412e-bc2c-0627e47df165)


[Live demo](https://lukehollis.github.io/three-mlagents/minecraft)

### Glider

![glider_example](https://github.com/user-attachments/assets/1c82281a-6982-4d05-95e4-816e6b8f61b4)


[Live demo](https://lukehollis.github.io/three-mlagents/glider)

### Astrodynamics

![astrodynamics_example](https://github.com/user-attachments/assets/d06b7d03-1ade-409f-aabc-fb1b283c7aab)

[Live demo](https://lukehollis.github.io/three-mlagents/astrodynamics)

### Fish

![fish_example](https://github.com/user-attachments/assets/2b45d7c0-7eac-4cc1-a383-9a92740d78e1)

[Live demo](https://lukehollis.github.io/three-mlagents/fish)


### Self-driving Car (Interpretability)

![self_driving_car_interpretability_example](https://github.com/user-attachments/assets/7bf64b31-71fa-4c3b-a872-9aa952df4285)


[Live demo](https://lukehollis.github.io/three-mlagents/self-driving-car)


### Labyrinth (NetHack)

![labyrinth_example](https://github.com/user-attachments/assets/39a2fe92-d279-4d23-a816-9c2a76e538b6)

[Live demo](https://lukehollis.github.io/three-mlagents/labyrinth)


| Demo                           | Live Demo                                                          |
|--------------------------------|--------------------------------------------------------------------|
| Basic 1-D Move to Goal         | [Live Demo](https://lukehollis.github.io/three-mlagents/basic)     |
| 3DBall Balance                 | [Live Demo](https://lukehollis.github.io/three-mlagents/ball3d)    |
| GridWorld Navigation           | [Live Demo](https://lukehollis.github.io/three-mlagents/gridworld) |
| Push-Block                     | [Live Demo](https://lukehollis.github.io/three-mlagents/push)      |
| Wall Jump                      | [Live Demo](https://lukehollis.github.io/three-mlagents/walljump)  |
| Ant (Crawler)                  | [Live Demo](https://lukehollis.github.io/three-mlagents/crawler)   |
| Swimmer / Worm                 | [Live Demo](https://lukehollis.github.io/three-mlagents/worm)      |
| Brick Break                    | [Live Demo](https://lukehollis.github.io/three-mlagents/brickbreak)|
| Food Collector                 | [Live Demo](https://lukehollis.github.io/three-mlagents/foodcollector) |
| Bicycle                        | [Live Demo](https://lukehollis.github.io/three-mlagents/bicycle)   |
| Intersection                   | [Live Demo](https://lukehollis.github.io/three-mlagents/intersection) |


## Three ML-Agents 

The library in Python+threejs should migrate easily matching from the previous ML-Agents if you're familiar with those examples.

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Vector Observations** | ✅ Stable | Standard numeric observations supported. |
| **Visual Observations** | ✅ Beta | `CameraSensor` implemented. Warning: Uses Base64/JSON (Prototype efficiency). |
| **Ray Perception** | ✅ Stable | `RayPerceptionSensor` simulates Lidar/Raycasts including tag detection. |
| **Side Channels** | ✅ Stable | Support for `EngineConfiguration` (timescale), `Stats` (logging), and `EnvironmentParameters`. |
| **Decision Requester** | ✅ Stable | Agents can request decisions at customized intervals (skip-frames). |

**Architecture Note**:
The migration uses bridge from Python to Javascript:
*   **Python**: `api/mlagents_bridge` (Standalone env compatible with `mlagents-envs`).
*   **JavaScript**: `client/src/libs/ml-agents` (Modular Agent/Academy/Sensor architecture).

---

## Project layout

```
three-mlagents/
│
├─ client/         ← Vite + React + Three.js front-end (examples live here)
│   ├─ src/examples
│   │   ├─ Index.jsx  ← landing page listing all examples
│   │   └─ Basic.jsx  ← 1-D "move-to-goal" environment matching Unity Basic
│   └─ src/libs/ml-agents  ← New JS Client SDK (Academy, Agent, Sensors)
│
├─ api/            ← FastAPI micro-service (Python ≥3.9)
│   ├─ mlagents_bridge/    ← New Python Environment Bridge
│   ├─ main.py      ← gym-style REST API for each environment
│   └─ requirements.txt
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


[![Star History Chart](https://api.star-history.com/svg?repos=lukehollis/three-mlagents&type=Date)](https://star-history.com/#lukehollis/three-mlagents&Date)

---

## License

MIT © 2025
