import React from 'react';
import { Grid, Text, useMediaQuery } from '@geist-ui/core';
import { Github } from '@geist-ui/icons';
import TronBackground from '../components/TronBackground.jsx';
import ExampleCard from '../components/ExampleCard.jsx';
import Layout from '../components/Layout.jsx';

const examples = [
  {
    title: "Glider",
    description: "Learn to perform dynamic soaring to stay aloft indefinitely in a wind gradient.",
    image: "/three-mlagents/glider_example.jpg",
    link: "/glider"
  },
  {
    title: "Minecraft (RL + LLM)",
    description: "Minecraft-like agents explore a world and mine resources, communicating their goals and crafting items.",
    image: "/three-mlagents/minefarm_example.jpg",
    link: "/minecraft"
  },
  {
    title: "Self-Driving Car",
    description: "A self-driving car learning to navigate a city and explaining its decisions.",
    image: "/three-mlagents/self_driving_car_interpretability_example.jpg",
    link: "/self-driving-car"
  },
  {
    title: "Astrodynamics",
    description: "Learn orbital rendezvous and docking with a space station using realistic orbital mechanics.",
    image: "/three-mlagents/astrodynamics_example.jpg",
    link: "/astrodynamics"
  },
  {
    title: "SimCity",
    description: "Use RL+LLM to play a SimCity-like simulation on a map with intelligent agents.",
    image: "/three-mlagents/simcity_example.jpg",
    link: "/simcity"
  },
  {
    title: "Fish",
    description: "A school of fish that learn to find food and survive a shark.",
    image: "/three-mlagents/fish_example.jpg",
    link: "/fish"
  },
  {
    title: "Simulants",
    description: "Build simulated worlds with intelligent agents, animals, and other characters.",
    image: "/three-mlagents/simulants_dev.jpg",
    link: "https://simulants.dev"
  },
  {
    title: "BrickBreak",
    description: "Destroy all the bricks with a ball and paddle.",
    image: "/three-mlagents/brick_break_example.jpg",
    link: "/brickbreak"
  },
  {
    title: "SimCity (Deck.gl)",
    description: "A Deck.gl-based visualization for the SimCity environment.",
    image: "/three-mlagents/simcity_deckgl_example.jpg",
    link: "/simcity-deckgl"
  },
  {
    title: "Intersection",
    description: "Simulate traffic at an intersection with traffic lights.",
    image: "/three-mlagents/intersection_example.jpg",
    link: "/intersection"
  },
  {
    title: "Bicycle",
    description: "Learn to ride a bicycle and keep it from falling over.",
    image: "/three-mlagents/bicycle_example.jpg",
    link: "/bicycle"
  },
  {
    title: "Labyrinth (NetHack)",
    description: "Theseus must escape the labyrinth before the Minotaur finds him.",
    image: "/three-mlagents/labyrinth_example.jpg",
    link: "/labyrinth"
  },
  {
    title: "Food Collector",
    description: "Multi-agent competition to collect good food and avoid bad food.",
    image: "/three-mlagents/food_collector_example.jpg",
    link: "/foodcollector"
  },
  {
    title: "Basic Environment",
    description: "1-D Move-To-Goal - Agent learns to reach targets for rewards.",
    image: "/three-mlagents/basic_example.jpg",
    link: "/basic"
  },
  {
    title: "3DBall Balance",
    description: "Tilt the platform to keep the ball from falling off the edge.",
    image: "/three-mlagents/3d_ball_example.jpg",
    link: "/ball3d"
  },
  {
    title: "GridWorld Navigation",
    description: "Navigate to the correct goal while avoiding incorrect ones.",
    image: "/three-mlagents/grid_world_example.jpg",
    link: "/gridworld"
  },
  {
    title: "Push-Block",
    description: "Push the box to the goal strip while learning optimal manoeuvres.",
    image: "/three-mlagents/push_block_example.jpg",
    link: "/push"
  },
  {
    title: "Wall Jump",
    description: "Learn to jump over walls to reach the goal.",
    image: "/three-mlagents/wall_jump_example.jpg",
    link: "/walljump"
  },
  {
    title: "Ant (Crawler)",
    description: "Move the ant towards the goal while maintaining balance.",
    image: "/three-mlagents/ant_example.jpg",
    link: "/crawler"
  },
  {
    title: "Worm",
    description: "Learn to swim and move towards a goal direction.",
    image: "/three-mlagents/worm_example.jpg",
    link: "/worm"
  }
];

export default function ExamplesIndex() {
  const isXs = useMediaQuery('xs');

  return (
    <>
      <TronBackground />

      <Layout>
        <div style={{
          backgroundColor: 'transparent',
          position: 'relative',
          zIndex: 1,
          padding: isXs ? '1.5rem 0 1rem' : '3rem 1rem 1rem',
          textAlign: 'center'
        }}>
          <Text h1 style={{
            color: '#fff',
            marginBottom: '1rem',
            textShadow: '0 0 20px rgba(255, 255, 255, 0.2)',
            fontSize: isXs ? '1.8rem' : '2.5rem',
            lineHeight: isXs ? '2.4rem' : '3rem',
            letterSpacing: '0.15em',
            fontWeight: 700,
            textTransform: 'uppercase',
            fontFamily: '"Orbitron", "Roboto Mono", monospace'
          }}>Three ML-Agents</Text>
          <Text p style={{ color: '#aaa', maxWidth: '600px', margin: 'auto', fontFamily: '"Roboto Mono", monospace', fontSize: '0.9rem' }}>
            Interactive reinforcement learning environments in the browser. Learn more about the project and fork on <a href="https://github.com/lukehollis/three-mlagents" target="_blank" rel="noopener noreferrer" style={{ color: '#fff', textDecoration: 'underline' }}>GitHub</a>. Built by <a href="https://github.com/lukehollis" target="_blank" rel="noopener noreferrer" style={{ color: '#fff', textDecoration: 'underline' }}>Luke Hollis</a>.
          </Text>
        </div>

        <div style={{ backgroundColor: 'transparent', position: 'relative', zIndex: 1 }}>
          <Grid.Container gap={isXs ? 1.5 : 2} justify="center">
            {examples.map((example) => (
              <Grid xs={24} sm={16} md={12} lg={8} key={example.title}>
                <ExampleCard {...example} />
              </Grid>
            ))}
          </Grid.Container>
        </div>
      </Layout>

      {/* Fixed "Fork on GitHub" link */}
      <a
        href="https://github.com/lukehollis/three-mlagents"
        target="_blank"
        rel="noopener noreferrer"
        style={{
          position: 'fixed',
          right: isXs ? '10px' : '20px',
          bottom: isXs ? '10px' : '20px',
          backgroundColor: '#000',
          color: '#fff',
          padding: isXs ? '8px 12px' : '10px 16px',
          borderRadius: 0,
          fontWeight: 600,
          textDecoration: 'none',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          border: '1px solid #fff',
          fontSize: '0.8rem',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          fontFamily: '"Roboto Mono", monospace'
        }}
      >
        <Github size={16} />
        Fork on GitHub
      </a>
    </>
  );
}