# Routine to store procedures for the sensitivity analysis that will be
# required in the optimization procedure of the training of ANNs and 
# PINNs models

# Defines a function to calculate the gradient using ForwardDiff automa-
# tic differentiation

function gradient_automaticDiff(f, variables)

    return ForwardDiff.gradient(f, variables)

end

# Defines a function to calculate the jacobian using ForwardDiff automa-
# tic differentiation

function jacobian_automaticDiff(f, variables)

    return ForwardDiff.jacobian(f, variables)

end

# Defines a function to calculate the hessian matrix using ForwardDiff

function hessian_automaticDiff(f, variables, f_dimensionality)

    # Sets a vector of hessians

    vector_hessians = Vector{Matrix{Float64}}(undef, f_dimensionality)

    # Iterates through the dimensionality of f

    for i=1:f_dimensionality

        # Sets the driver

        g(variables) = f(variables)[i]

        # Calculates the hessian

        vector_hessians[i] = ForwardDiff.hessian(g, variables)

    end

    return vector_hessians

end

# Defines a function to evaluate the gradient of a scalar function using
# central finite differences

function gradient_CFD(f::Function, variables::Vector{Float64}; epsilon=
 1E-5)

    # Initializes the gradient vector

    df = zeros(length(variables))

    # Iterates through the variables

    for i=1:length(variables)

        # Gets the variables vector ahead

        x_ahead = copy(variables)

        x_ahead[i] += epsilon

        # Evaluates the function ahead

        f_ahead = f(x_ahead)

        # Gets the variables vector abaft

        x_abaft = copy(variables)

        x_abaft[i] -= epsilon

        # Evaluates the function abaft

        f_abaft = f(x_abaft)

        # Calculates the derivative

        df[i] = ((f_ahead-f_abaft)/(2*epsilon))

    end 

    # Returns the gradient

    return df

end