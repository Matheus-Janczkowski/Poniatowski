# Routine to store different activations functions

# Sigmoid

sigmoid(x) = (1/(1+exp(-x)))

# Hyperbolic tangent

tanh(x) = ((2*sigmoid(2*x))-1)

# Linear

linear(x) = x

# Leaky-relu

function leaky_relu(x; alpha=0.1)

    if x>0

        return x 

    else 

        return x*alpha

    end 

end