# Routine to store procedures for the sensitivity analysis that will be
# required in the optimization procedure of the training of ANNs and 
# PINNs models

# Defines a function to calculate the gradient using Zygote's automatic
# differentiation

function gradient_automaticDiff(f, params_vector)

    return gradient(f, params_vector)[1]

end