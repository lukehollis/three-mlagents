import React from 'react';
import { Link } from 'react-router-dom';
import { Page, Grid, Card, Text, Spacer, Button } from '@geist-ui/core';
import { Play } from 'geist-icons';
import TronBackground from '../components/TronBackground.jsx';

export default function ExamplesIndex() {
  return (
    <>
      <TronBackground />

      <Page style={{ backgroundColor: 'transparent', minHeight: '100vh' }}>
        <Page.Header style={{ backgroundColor: 'transparent', position: 'relative', zIndex: 1, padding: '3rem 0 1rem 0' }}>
          <Text h1 style={{ color: '#fff', marginBottom: '8px', textShadow: '0 0 10px rgba(0, 255, 255, 0.5)' }}>Three ML-Agents Examples</Text>
          <Text p style={{ color: '#ccc' }}>Interactive reinforcement learning environments in the browser. Learn more about the project and fork on <a href="https://github.com/lukehollis/three-mlagents" target="_blank" rel="noopener noreferrer" style={{ color: '#00ffff', textDecoration: 'none' }}>GitHub</a>.</Text>
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
              }}
            >
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
        </Grid.Container>
        
              </Page.Content>
      </Page>
    </>
  );
} 