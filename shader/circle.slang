#language slang 2026

uniform float frame;
uniform float2 screenSize;

typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;

// Output buffer: flatten the 2D screen to a 1D array
RWStructuredBuffer<float4> output : register(u0);

[shader("compute")]
[numthreads(16, 16, 1)]
void imageMain(uint2 dispatchThreadID: SV_DispatchThreadID)
{
    const float PI = 3.14159265359;
    
    vec2 p = (dispatchThreadID.xy * 2.0f - screenSize.xy) / (float)screenSize.y;
    float tau = PI * 2.0;
    float a = atan2(p.x,p.y);
    float r = length(p)*0.75;
    vec2 uv = vec2(a / tau, r);
    float t = frame / 60.f;

    // get the color
    float xCol = (uv.x - t / 3.0) * 3.0;
    xCol = fmod(abs(xCol), 3.0f);
    vec3 horColour = vec3(0.25, 0.25, 0.25);
		
    if (xCol < 1.0)
    {
        horColour.r += 1.0 - xCol;
        horColour.g += xCol;
    }
    else if (xCol < 2.0)
    {
        xCol -= 1.0;
        horColour.g += 1.0 - xCol;
        horColour.b += xCol;
    }
    else
    {
        xCol -= 2.0;
        horColour.b += 1.0 - xCol;
        horColour.r += xCol;
    }

    // draw color beam
    uv = (2.0 * uv) - 1.0;
    float beamWidth = (0.7 + 
        0.5 * cos(uv.x * 10.0 * tau * 0.15 * clamp(floor(5.0 + 10.0 * cos(t)), 0.0, 10.0)))
        * abs(1.0 / (30.0 * uv.y));
    vec3 horBeam = vec3(beamWidth);
    vec4 color = vec4(horBeam * horColour, 1.0);

    // Flatten 2D coordinates into 1D
    uint index = dispatchThreadID.y * uint(screenSize.x) + dispatchThreadID.x;
    output[index] = color;
}
