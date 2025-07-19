import React, { useRef, useEffect } from 'react';
import { Card, Text } from '@geist-ui/core';

const CameraFeed = ({ feedData, name }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        if (feedData && canvasRef.current) {
            const canvas = canvasRef.current;
            const context = canvas.getContext('2d');
            const { buffer, width, height } = feedData;
            
            const imageData = new ImageData(new Uint8ClampedArray(buffer.buffer), width, height);

            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            const tempContext = tempCanvas.getContext('2d');
            tempContext.putImageData(imageData, 0, 0);

            context.clearRect(0, 0, canvas.width, canvas.height);
            context.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
        }
    }, [feedData]);


    return (
        <div 
            style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'start',
                justifyContent: 'start',
            }}
        >
            <Text style={{ 
                margin: '0 0 5px 0', 
                fontSize: '10px', color: '#fff', background: 'rgba(0,0,0,0.6)', padding: '4px', borderRadius: '4px', width: 'auto', textTransform: 'uppercase', display: 'inline-block' 
                }}
            >
                {name}
            </Text>
            <canvas 
                ref={canvasRef} 
                width="120"
                height="90" 
                style={{ 
                    border: '1px solid #555', borderRadius: '4px', 
                    width: '120px',
                    background: 'rgba(0,0,0,0.6)',
                }} 
            />
        </div>
    );
};


const CameraFeeds = ({ cameraFeedData }) => {
    if (!cameraFeedData || Object.keys(cameraFeedData).length === 0) {
        return null;
    }

    return (
        <div
            style={{
                position: 'absolute', 
                top: '180px', 
                left: '10px',
                width: '260px',
                color: '#fff',
                display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px'
            }}
        >
            {Object.entries(cameraFeedData).map(([name, data]) => (
                <CameraFeed key={name} name={name.replace(/^\d+-/, '')} feedData={data} />
            ))}
        </div>
    );
};

export default CameraFeeds; 