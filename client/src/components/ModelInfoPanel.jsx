import React from 'react';
import { Card, Text } from '@geist-ui/core';

const ModelInfoPanel = ({ modelInfo }) => {
    if (!modelInfo) return null;

    return (
        <Card style={{
            position: 'absolute',
            bottom: '10px',
            right: '10px',
            width: '250px',
            background: 'rgba(0,0,0,0.6)',
            color: '#fff',
            border: '1px solid #444',
        }}>
            <Text h5 style={{ margin: 0, color: '#fff' }}>Trained Model Info</Text>
            <Text p style={{ margin: '4px 0', fontSize: '12px', opacity: 0.8 }}>
                Epochs: {modelInfo.epochs}
            </Text>
            <Text p style={{ margin: '4px 0', fontSize: '12px', opacity: 0.8 }}>
                Final Loss: {modelInfo.loss.toFixed(4)}
            </Text>
        </Card>
    );
};

export default ModelInfoPanel; 