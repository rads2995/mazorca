// Example slang shader to be compiled at run-time

import autodiff;

// hello_world.slang
StructuredBuffer<float> buffer0;
StructuredBuffer<float> buffer1;
RWStructuredBuffer<float> result;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    DifferentialPair<float> dp_a = diffPair(
        1.0, // Input 'a'
        1.0  // Vector 'v' for vector-Jacobian product input (for 'a')
    );
    
    DifferentialPair<float> dp_b = diffPair(2.4, 0.0);

    // fwd_diff to compue output and d_output w.r.t 'a'
    // Our output is also a differential pair
    DifferentialPair<float2> dp_output = fwd_diff(foo)(dp_a, dp_b);

    // Extract output's primal part, which is the standard output when foo is called normal
    float2 output_p = dp_output.getPrimal();

    // Extract output's derivative part
    float2 output_d = dp_output.getDifferential();

    uint index = threadId.x;
    result[index] = buffer0[index] + buffer1[index];
}
