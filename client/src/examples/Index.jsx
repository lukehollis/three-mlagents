import React from 'react';
import { Link } from 'react-router-dom';
import { Page, Grid, Card, Text, Spacer, Button } from '@geist-ui/core';
import { Play } from 'geist-icons';
import { Github } from '@geist-ui/icons';
import TronBackground from '../components/TronBackground.jsx';

export default function ExamplesIndex() {
  return (
    <>
      <TronBackground />

      <Page style={{ backgroundColor: 'transparent', minHeight: '100vh' }}>
        <Page.Header style={{ backgroundColor: 'transparent', position: 'relative', zIndex: 1, padding: '3rem 0 1rem 0' }}>
          <Text h1 style={{ color: '#fff', marginBottom: '8px', textShadow: '0 0 10px rgba(0, 255, 255, 0.5)' }}>Three ML-Agents Examples</Text>
          <Text p style={{ color: '#ccc' }}>Interactive reinforcement learning environments in the browser. Learn more about the project and fork on <a href="https://github.com/lukehollis/three-mlagents" target="_blank" rel="noopener noreferrer" style={{ color: '#00ffff', textDecoration: 'none' }}>GitHub</a>. Built by <a href="https://github.com/lukehollis" target="_blank" rel="noopener noreferrer" style={{ color: '#00ffff', textDecoration: 'none' }}>Luke Hollis</a>.</Text>
        </Page.Header>
        
        <Page.Content style={{ backgroundColor: 'transparent', position: 'relative', zIndex: 1 }}>
        <Grid.Container gap={2} justify="center">
          <Grid xs={24} sm={16} md={12} lg={8}>
            <Card 
              hoverable 
              style={{ 
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
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/basic" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer' }}>
                    Basic Environment
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5' }}>
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
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/ball3d" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer' }}>
                    3DBall Balance
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5' }}>
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
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/gridworld" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer' }}>
                    GridWorld Navigation
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5' }}>
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
                      src="/three-mlagents/3d_ball_example.jpg" 
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
              <Card.Footer style={{ backgroundColor: 'rgba(17, 17, 17, 0.9)', padding: '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <Link to="/push" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <Text h4 style={{ color: '#fff', margin: '0 0 8px 0', cursor: 'pointer' }}>
                    Push-Block
                  </Text>
                </Link>
                <Text p style={{ color: '#888', margin: '0 0 16px 0', lineHeight: '1.5' }}>
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
          right: '20px',
          bottom: '20px',
          backgroundColor: 'rgba(17, 17, 17, 0.8)',
          color: '#fff',
          padding: '10px 16px',
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