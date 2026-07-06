import logging
import unittest
import asyncio

import httpx
import numpy as np

from main import app

from mlagents.registry import get_task, list_tasks, make_env
from mlagents.training import (
    ALGORITHMS,
    POLICIES_DIR,
    _default_model_kwargs,
    _default_policy,
    _resolve_model_path,
    make_vector_env,
    predict_action,
)

logging.getLogger("httpx").setLevel(logging.WARNING)


class RegistryTests(unittest.TestCase):
    def test_trainable_tasks_have_factories(self):
        trainable = list_tasks(include_roadmap=False)
        self.assertGreaterEqual(len(trainable), 5)
        for task in trainable:
            self.assertTrue(task.trainable, task.id)
            self.assertEqual(task.card()["trainable"], True)

    def test_basic_env_reset_and_step(self):
        env = make_env("basic")
        try:
            obs, info = env.reset(seed=1)
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertEqual(info["position"], 10)
            next_obs, reward, terminated, truncated, info = env.step(2)
            self.assertEqual(next_obs.shape, env.observation_space.shape)
            self.assertIsInstance(reward, float)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["position"], 11)
        finally:
            env.close()

    def test_alias_resolution(self):
        self.assertEqual(get_task("brick-break").id, "brickbreak")
        self.assertEqual(get_task("self_driving_car").id, "self-driving-car")

    def test_trainable_env_contracts_match_declared_spaces(self):
        for task in list_tasks(include_roadmap=False):
            with self.subTest(task=task.id):
                env = make_env(task.id)
                try:
                    obs, _ = env.reset(seed=123)
                    self.assertTrue(
                        env.observation_space.contains(obs),
                        f"{task.id} reset obs violates {env.observation_space}",
                    )
                    next_obs, reward, terminated, truncated, _ = env.step(
                        env.action_space.sample()
                    )
                    self.assertTrue(
                        env.observation_space.contains(next_obs),
                        f"{task.id} step obs violates {env.observation_space}",
                    )
                    self.assertIsInstance(float(reward), float)
                    self.assertIsInstance(terminated, bool)
                    self.assertIsInstance(truncated, bool)
                finally:
                    env.close()

    def test_registered_sb3_algorithms_construct_and_predict(self):
        for task in list_tasks(include_roadmap=False):
            with self.subTest(task=task.id):
                n_envs = (
                    1
                    if task.default_algorithm in {"dqn", "sac", "td3"}
                    else min(2, task.n_envs)
                )
                vec_env = make_vector_env(task.id, n_envs=n_envs, seed=321)
                try:
                    kwargs = _default_model_kwargs(
                        task.default_algorithm,
                        train_env=vec_env,
                        task=task,
                        total_timesteps=64,
                        tensorboard_log="/tmp/three-mlagents-test-tb",
                        verbose=0,
                    )
                    model = ALGORITHMS[task.default_algorithm](
                        _default_policy(task),
                        vec_env,
                        seed=321,
                        **kwargs,
                    )
                    action, _ = model.predict(vec_env.reset(), deterministic=True)
                    self.assertIsNotNone(action)
                finally:
                    vec_env.close()


class PredictionShapeTests(unittest.TestCase):
    def test_predict_requires_model_file(self):
        obs = np.zeros(21, dtype=np.float32)
        with self.assertRaises(FileNotFoundError):
            predict_action("basic", obs, "missing.zip")

    def test_model_path_resolver_accepts_policy_relative_path(self):
        task = get_task("basic")
        model_path = POLICIES_DIR / "resolver_regression_test.zip"
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"placeholder")
            self.assertEqual(_resolve_model_path(task, str(model_path)), model_path)
            self.assertEqual(
                _resolve_model_path(task, model_path.name),
                model_path,
            )
        finally:
            model_path.unlink(missing_ok=True)


class ApiRouteTests(unittest.TestCase):
    def test_task_metadata_routes(self):
        async def _request_routes():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return (
                    await client.get("/health"),
                    await client.get("/tasks"),
                    await client.get("/tasks/basic"),
                )

        health, tasks, basic = asyncio.run(_request_routes())

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        self.assertEqual(tasks.status_code, 200)
        task_cards = tasks.json()["tasks"]
        self.assertGreaterEqual(len(task_cards), 13)

        self.assertEqual(basic.status_code, 200)
        self.assertEqual(basic.json()["id"], "basic")
        self.assertTrue(basic.json()["trainable"])


if __name__ == "__main__":
    unittest.main()
