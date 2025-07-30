import React from 'react';
import { Link } from 'react-router-dom';
import { Page, Grid, Card, Text, Spacer, Button, useMediaQuery } from '@geist-ui/core';
import { Play } from 'geist-icons';
import { Github } from '@geist-ui/icons';
import TronBackground from '../components/TronBackground.jsx';

export default function ExamplesIndex() {
  const isXs = useMediaQuery('xs');
  return (
    <>
      <TronBackground />

      <Page style={{ backgroundColor: 'transparent', minHeight: '100vh', padding: isXs ? '0 1rem' : undefined, margin: '0 auto', maxWidth: '1200px', width: '100vw' }}>
        <Page.Header style={{ backgroundColor: 'transparent', position: 'relative', zIndex: 1, padding: isXs ? '1.5rem 0 1rem' : '3rem 1rem 1rem', textAlign: 'center' }}>
          <Text h1 style={{ 
            color: '#fff', 
            marginBottom: '1rem', 
            textShadow: '0 0 10px rgba(0, 255, 255, 0.5)',
            fontSize: isXs ? '2.2rem' : '3rem',
            lineHeight: isXs ? '2.8rem' : '3.5rem',
          }}>Three ML-Agents Examples</Text>
          <Text p style={{ color: '#ccc', maxWidth: '600px', margin: 'auto' }}>Interactive reinforcement learning environments in the browser. Learn more about the project and fork on <a href="https://github.com/lukehollis/three-mlagents" target="_blank" rel="noopener noreferrer" style={{ color: '#00ffff', textDecoration: 'none' }}>GitHub</a>. Built by <a href="https://github.com/lukehollis" target="_blank" rel="noopener noreferrer" style={{ color: '#00ffff', textDecoration: 'none' }}>Luke Hollis</a>.</Text>
        </Page.Header>
        
        <Page.Content style={{ backgroundColor: 'transparent', position: 'relative', zIndex: 1 }}>
        <Grid.Container gap={isXs ? 1.5 : 2} justify="center">
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/glider" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/glider_example.jpg" 
                      alt="Glider Dynamic Soaring Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/glider" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Glider
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Learn to perform dynamic soaring to stay aloft indefinitely in a wind gradient.
                </Text>
                <Link to="/glider" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/minecraft" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/minefarm_example.jpg" 
                      alt="Minecraft "
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/minecraft" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Minecraft (RL + LLM)
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Minecraft-like agents explore a world and mine resources, communicating their goals and working together to craft items using a language model. 
                </Text>
                <Link to="/minecraft" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/self-driving-car" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/self_driving_car_interpretability_example.jpg" 
                      alt="Self-Driving Car Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/self-driving-car" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Self-Driving Car (Interpretability)
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  A self-driving car learning to navigate a city and explaining its decisions.
                </Text>
                <Link to="/self-driving-car" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/simcity" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/simcity_example.jpg" 
                      alt="SimCity Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/simcity" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    SimCity
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Use RL+LLM to play a SimCity-like simulation on a map with intelligent agents.
                </Text>
                <Link to="/simcity" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/simcity-kepler" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/simcity_example.jpg" 
                      alt="SimCity Kepler Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/simcity-kepler" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    SimCity Kepler
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Collaborative city building simulation with Kepler.gl map visualization and intelligent agents.
                </Text>
                <Link to="/simcity-kepler" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/fish" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/fish_example.jpg" 
                      alt="Fish Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/fish" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Fish
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  A school of fish that learn to find food and survive a shark.
                </Text>
                <Link to="/fish" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/astrodynamics" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/astrodynamics_example.jpg" 
                      alt="Astrodynamics Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/astrodynamics" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Astrodynamics
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Learn orbital rendezvous and docking with a space station using realistic orbital mechanics and Hill's equations.
                </Text>
                <Link to="/astrodynamics" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/brickbreak" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/brick_break_example.jpg" 
                      alt="BrickBreak Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/brickbreak" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    BrickBreak
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Destroy all the bricks with a ball and paddle.
                </Text>
                <Link to="/brickbreak" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/intersection" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/intersection_example.jpg" 
                      alt="Intersection Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/intersection" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Intersection
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Simulate traffic at an intersection with traffic lights.
                </Text>
                <Link to="/intersection" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
           <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/bicycle" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/bicycle_example.jpg" 
                      alt="Bicycle Balance Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/bicycle" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Bicycle
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Learn to ride a bicycle and keep it from falling over.
                </Text>
                <Link to="/bicycle" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/labyrinth" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/labyrinth_example.jpg" 
                      alt="Labyrinth Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/labyrinth" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Labyrinth
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Theseus must escape the labyrinth before the Minotaur finds him.
                </Text>
                <Link to="/labyrinth" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
         <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/foodcollector" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/food_collector_example.jpg" 
                      alt="Food Collector Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/foodcollector" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Food Collector
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Multi-agent competition to collect good food and avoid bad food.
                </Text>
                <Link to="/foodcollector" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
                        </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>    
                <Link to="/basic" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/basic_example.jpg" 
                      alt="Basic 1-D Move-To-Goal Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/basic" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Basic Environment
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  1-D Move-To-Goal - Agent learns to reach targets for rewards
                </Text>
                <Link to="/basic" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/ball3d" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/3d_ball_example.jpg" 
                      alt="3DBall Balance Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/ball3d" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    3DBall Balance
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Tilt the platform to keep the ball from falling off the edge.
                </Text>
                <Link to="/ball3d" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/gridworld" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/grid_world_example.jpg" 
                      alt="GridWorld Navigation Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/gridworld" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    GridWorld Navigation
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Navigate to the correct goal while avoiding incorrect ones.
                </Text>
                <Link to="/gridworld" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/push" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/push_block_example.jpg" 
                      alt="Push-Block Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/push" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Push-Block
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Push the box to the goal strip while learning optimal manoeuvres.
                </Text>
                <Link to="/push" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/walljump" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/wall_jump_example.jpg" 
                      alt="Wall Jump Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/walljump" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Wall Jump
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Learn to jump over walls to reach the goal and avoid other obstacles in your path.
                </Text>
                <Link to="/walljump" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/crawler" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/ant_example.jpg" 
                      alt="Crawler Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/crawler" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Ant (Crawler)
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Move the ant towards the goal while maintaining balance and direction. (Ant-v5)
                </Text>
                <Link to="/crawler" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
                width: '100%',
                backgroundColor: 'rgba(17, 17, 17, 0.8)', 
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                overflow: 'hidden'
              }}
            >
              <Card.Content style={{ padding: 0 }}>
                <Link to="/worm" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div style={{ cursor: 'pointer' }}>
                    <img 
                      src="/three-mlagents/worm_example.jpg" 
                      alt="Worm Example"
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                  </div>
                </Link>
              </Card.Content>
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/worm" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer', fontSize: isXs ? '1.1rem' : '1.25rem' }}>
                    Worm
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5', fontSize: isXs ? '0.875rem' : '1rem' }}>
                  Learn to swim and move towards a goal direction. (Swimmer-v5)
                </Text>
                <Link to="/worm" style={{ textDecoration: 'none' }}>
                  <Button 
                    type="success" 
                    icon={<Play />} 
                    auto
                  >
                    Launch Example
                  </Button>
                </Link>
              </Card.Footer>
            </Card>
          </Grid>
        </Grid.Container>
        </Page.Content>
      </Page>

      {/* Fixed "Fork on GitHub" link */}
      <a
        href="https://github.com/lukehollis/three-mlagents"
        target="_blank"
        rel="noopener noreferrer"
        style={{
          position: 'fixed',
          right: isXs ? '10px' : '20px',
          bottom: isXs ? '10px' : '20px',
          backgroundColor: 'rgba(17, 17, 17, 0.8)',
          color: '#fff',
          padding: isXs ? '8px 12px' : '10px 16px',
          borderRadius: '6px',
          fontWeight: 500,
          textDecoration: 'none',
          backdropFilter: 'blur(8px)',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          border: '1px solid #fff',
        }}
      >
        <Github size={12} style={{ border: '1px solid #fff' }} />
        Fork on GitHub
      </a>
    </>
  );
} 