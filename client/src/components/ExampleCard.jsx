import { Link } from 'react-router-dom';
import { Card, Text, Button, useMediaQuery } from '@geist-ui/core';

export default function ExampleCard({ title, description, image, link, buttonText = "Launch Example" }) {
  const isXs = useMediaQuery('xs');

  return (
    <Card 
      hoverable 
      style={{ 
        width: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.8)', 
        border: '1px solid #333',
        borderRadius: 0,
        boxShadow: 'none',
        transition: 'all 0.3s ease',
        overflow: 'visible', // Allow markers to hang out if needed, or keep hidden if inner
        position: 'relative'
      }}
    >

      <Card.Content style={{ padding: 0, borderRadius: 0 }}>
        <Link to={link} style={{ textDecoration: 'none', color: 'inherit' }}>
          <div style={{ cursor: 'pointer', position: 'relative' }}>
            <img 
              src={image} 
              alt={title}
              style={{
                width: '100%',
                height: '200px',
                objectFit: 'cover',
                display: 'block',
                borderRadius: 0
              }}
            />
            {/* Image Overlay Grid Effect (Optional Scanline) */}
            <div style={{ 
              position: 'absolute', 
              top: 0, 
              left: 0, 
              right: 0, 
              bottom: 0, 
              background: 'linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))',
              backgroundSize: '100% 2px, 3px 100%',
              pointerEvents: 'none'
            }} />
          </div>
        </Link>
      </Card.Content>
      <Card.Footer style={{ backgroundColor: '#000', padding: isXs ? '12px' : '16px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start', borderTop: '1px solid #333', borderRadius: 0 }}>
        <Link to={link} style={{ textDecoration: 'none', color: 'inherit' }}>
          <Text h4 style={{ 
            color: '#fff', 
            margin: '0 0 8px 0', 
            cursor: 'pointer', 
            fontSize: isXs ? '1rem' : '1.1rem',
            fontFamily: '"Orbitron", "Roboto Mono", monospace',
            textTransform: 'uppercase',
            letterSpacing: '0.05em'
          }}>
            {title}
          </Text>
        </Link>
        <Text p style={{ 
          color: '#aaa', 
          margin: '0 0 16px 0', 
          lineHeight: '1.5', 
          fontSize: isXs ? '0.75rem' : '0.85rem',
          fontFamily: '"Roboto Mono", monospace' 
        }}>
          {description}
        </Text>
        <Link to={link} style={{ textDecoration: 'none', width: '100%' }}>
          <Button 
            width="100%"
            style={{ 
              backgroundColor: 'transparent', 
              color: '#fff', 
              border: '1px solid #fff', 
              borderRadius: 0,
              textTransform: 'uppercase', 
              fontSize: '0.75rem', 
              letterSpacing: '2px',
              fontWeight: 600,
              fontFamily: 'monospace'
            }}
          >
            {buttonText} {">>"}
          </Button>
        </Link>
      </Card.Footer>
    </Card>
  );
}
