// reference: https://www.shadertoy.com/view/tfByRK

const int   MAX_RAY_STEPS   = 140;
const float TAIL_LENGTH     = 10.5; 
const float TAIL_DECAY      = 0.5;  
const float FALL_SPEED      = 12.0; 
const float GLOW_EXPOSURE   = 99.0; 
const float STAR_SIZE       = 1.5;  // Size of cross-flare-head
const float STAR_BRIGHTNESS = 2.0;  // Brightness of cross-flare-head

float sRGBencode(float linearColor) 
{ 
    return linearColor > 0.0031308 ? (1.055 * pow(linearColor, 1.0 / 2.4) - 0.055) : (12.92 * linearColor); 
}

vec3 sRGBencode(vec3 linearColor) 
{ 
    linearColor = clamp(linearColor, 0.0, 1.0); 
    return vec3(sRGBencode(linearColor.x), sRGBencode(linearColor.y), sRGBencode(linearColor.z)); 
}

float Star(vec2 uv, float flare)
{
    float d = length(uv);
    float m = 0.02 / (d + 0.001); // Add tiny number for dividing zero
    float rays = max(0.0, 1.0 - abs(uv.x * uv.y * 200.0)); 
    m += rays * flare;
    m *= smoothstep(0.8, 0.05, d); 
    return m;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 finalColor = vec3(0.0);
    float currentTime = iTime;

    // Camera Setup
    float focalLength = 2.0;
    vec3 rayOrigin = vec3(0.5, 3.8, 0.5 - currentTime); 
    vec3 rayDir = normalize(vec3(uv, -focalLength));    

    // Camera Rotation Angles
    float pitchAngle = -0.35;
    float cp = cos(pitchAngle), sp = sin(pitchAngle);
    float yawAngle = 0.25 * sin(currentTime * 0.3);
    float cy = cos(yawAngle), sy = sin(yawAngle);

    // Apply Rotation to Ray Direction
    rayDir.yz *= mat2(cp, sp, -sp, cp);
    rayDir.xz *= mat2(cy, sy, -sy, cy);

    // Calc Camera local axis(right, up) for 3D billboard
    vec3 camRight = vec3(1.0, 0.0, 0.0);
    camRight.yz *= mat2(cp, sp, -sp, cp);
    camRight.xz *= mat2(cy, sy, -sy, cy);

    vec3 camUp = vec3(0.0, 1.0, 0.0);
    camUp.yz *= mat2(cp, sp, -sp, cp);
    camUp.xz *= mat2(cy, sy, -sy, cy);

    // Raymarching Loop
    float rayDistance = 0.0;    
    for(int i = 0; i < MAX_RAY_STEPS; i++)
    {
        vec3 rayPos = rayDir * rayDistance + rayOrigin;
        vec2 gridCell = round(rayPos.xz);
        
        // Change for texture image size
        //ivec2 texCoord = (ivec2(gridCell) % 1024 + 1024) % 1024;
        //vec4 noiseData = texelFetch(iChannel0, texCoord, 0);       
        ivec2 texSize = textureSize(iChannel0, 0);
        ivec2 texCoord = (ivec2(gridCell) % texSize + texSize) % texSize;
        vec4 noiseData = texelFetch(iChannel0, texCoord, 0);

        float dropHeadY = mod(noiseData.a * 93.0 - currentTime * FALL_SPEED, 45.0);
        
        // Calc tail
        vec3 nearestPointOnLine = vec3(gridCell.x, clamp(rayPos.y, dropHeadY, dropHeadY + TAIL_LENGTH), gridCell.y);
        float distToLine = length(rayPos - nearestPointOnLine);
        
        float tailAttenuation = exp(-max(TAIL_DECAY * (rayPos.y - dropHeadY), 0.0)); 
        float tailGlow = tailAttenuation / (abs(distToLine * distToLine) + 0.001); 

        // Calc cross-flare-head position for billboard
        vec3 headPos = vec3(gridCell.x, dropHeadY, gridCell.y);
        vec3 deltaToHead = rayPos - headPos;

        // Project cross-shaped-head from 3d to 2d for billboard
        vec2 flareUV = vec2(dot(deltaToHead, camRight), dot(deltaToHead, camUp));
        
        // Rotate head through time
        float rotSpeed = currentTime * (3.0 + noiseData.b * 2.0);
        float cr = cos(rotSpeed), sr = sin(rotSpeed);
        flareUV *= mat2(cr, sr, -sr, cr);
        
        float headGlow = Star(flareUV * (1.0 / STAR_SIZE), STAR_BRIGHTNESS);
        float depthAttenuation = exp(-0.05 * rayDistance);        
        finalColor += depthAttenuation * (tailGlow * 0.5 + headGlow * 15.0) * noiseData.rgb;
        
        rayDistance += min(0.5, distToLine) + 0.01;
    }    

    finalColor = tanh(finalColor / GLOW_EXPOSURE);
    finalColor = sRGBencode(finalColor);
    fragColor = vec4(finalColor, 1.0);
}
