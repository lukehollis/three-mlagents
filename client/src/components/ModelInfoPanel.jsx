import React from 'react';
import { Card, Text } from '@geist-ui/core';

const ModelInfoPanel = ({ modelInfo }) => {
    if (!modelInfo) return null;

    return (
        <Card style={{
            position: 'fixed',
            top: '160px',
            left: '10px',
            width: '250px',
            background: 'rgba(0,0,0,0.8)',
            color: '#fff',
            border: '1px solid #333',
            borderRadius: 0,
            display: 'none',
            fontFamily: 'monospace'
        }}>
            {/* Tactical Markers */}
             <div style={{ position: 'absolute', top: 0, left: 0, width: '6px', height: '6px', borderTop: '1px solid #fff', borderLeft: '1px solid #fff' }} />
             <div style={{ position: 'absolute', top: 0, right: 0, width: '6px', height: '6px', borderTop: '1px solid #fff', borderRight: '1px solid #fff' }} />
             <div style={{ position: 'absolute', bottom: 0, left: 0, width: '6px', height: '6px', borderBottom: '1px solid #fff', borderLeft: '1px solid #fff' }} />
             <div style={{ position: 'absolute', bottom: 0, right: 0, width: '6px', height: '6px', borderBottom: '1px solid #fff', borderRight: '1px solid #fff' }} />
            <Text h5 style={{ margin: 0, color: '#fff' }}>Trained Model Info</Text>
            <Text p style={{ margin: '4px 0', fontSize: '12px', opacity: 0.8 }}>
                Epochs: {modelInfo.epochs}
            </Text>
            <Text p style={{ margin: '4px 0', fontSize: '12px', opacity: 0.8 }}>
                Final Loss: {typeof modelInfo.loss === 'number' ? modelInfo.loss.toFixed(4) : modelInfo.loss}
            </Text>
        </Card>
    );
};

export default ModelInfoPanel; 