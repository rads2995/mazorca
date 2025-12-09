[Differentiable] // Auto-diff requires that functions are marked differentiable
float2 foo(float a, float b) {
    return float2(a * b * b, a * a);
}
