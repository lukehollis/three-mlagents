---
name: roadmap-marl
description: Plan multi-agent and simulator-heavy environments using appropriate APIs instead of forcing them into single-agent SB3.
argument-hint: "[task-id]"
allowed-tools: Read, Write, Glob, Bash
context: repo
agent: three-mlagents-roadmap-marl-planner
---

# Roadmap MARL Skill

Use this skill for multi-agent games, traffic, ecological systems,
self-driving, social simulation, LLM agents, or external simulator bridges.

## Interface Decision

Choose the interface before implementation:

- Gymnasium: one learning agent controlling one policy.
- PettingZoo AEC: turn-based multi-agent environments.
- PettingZoo ParallelEnv: simultaneous-action multi-agent environments.
- External bridge: simulator owns stepping and exposes a documented protocol.
- Roadmap visualization: frontend demo exists, but training is not yet a
  scientific baseline.

## MARL Requirements

Before marking a task trainable, specify:

- agent ids and lifetimes;
- per-agent observation and action spaces;
- simultaneous versus sequential action semantics;
- reward ownership and global/team rewards;
- termination and truncation per agent;
- wrappers, vectorization, and baseline learner;
- evaluation metrics beyond aggregate reward;
- reproducible seeds and scenario generation.

## Baseline Guidance

Use PettingZoo-compatible tooling for true MARL. SuperSuit wrappers may be
useful for preprocessing/vectorization. For serious baselines, document whether
the plan uses IPPO, MAPPO, RLlib, CleanRL-style scripts, or another maintained
open implementation.

Do not quietly convert a MARL problem into a single-agent SB3 task. If a
centralized controller is scientifically intentional, name it as such in the
task card, UI, and evaluation notes.

## Existing Roadmap Tasks

Treat these as roadmap unless and until their backend environment and baseline
pipeline are implemented:

- `foodcollector`
- `fish`
- `intersection`
- `minecraft`
- `simcity`
- `self-driving-car`
