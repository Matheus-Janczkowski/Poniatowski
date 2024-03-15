# Routines to store loss functions for ANNs and PINNs. The loss func-
# tions ahead receive a vector of values and operate over their compo-
# nents, e.g., they can receive the vector of the differences between 
# the true and the predicted values in case of ANNs or the vector of re-
# sidue in the collocation points in case of PINNs

# Defines a function for the sum of squares

function mse(x)#x::Vector{Number})

    # Returns the sum of the components squared divided by the number of
    # components

    return (sum(number_squared.(x))/length(x))

end

# Defines a function for the root mean squared error

function rmse(x)#x::Vector{Number})

    # Returns the square root of the mse

    return sqrt(mse(x))

end

########################################################################
#                              Operations                              #
########################################################################

# Defines a function for the square of a number

function number_squared(x)#x::Number)

    return x^2

end
