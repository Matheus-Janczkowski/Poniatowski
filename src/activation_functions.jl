# Routine to store different activations functions

# Sigmoid

sigmoid(x) = (1/(1+exp(-x)))

# Hyperbolic tangent

tanh(x) = ((2*sigmoid(2*x))-1)

# Linear

linear_function(x; a1=1.0, a0=0.0) = (a1*x)+a0

# Quadratic

quadratic(x; a2=1.0, a1=0.0, a0=0.0) = (a2*(x^2))+(a1*x)+a0

# Leaky-relu

function leaky_relu(x; alpha=0.1)

    if x>0

        return x 

    else 

        return x*alpha

    end 

end