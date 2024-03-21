# Routine to evaluate artificial neural networks (ANNs)

########################################################################
#                              Evaluation                              #
########################################################################

# Receiving the parameters as vectors of matrices (weights) and of vec-
# tors (biases)

# Defines a function to evaluate the response of an ANN.
# The arguments are:
# X               - ANN's input vector;
# W               - Vector of weights' matrices. W has the weights for a
#                   neuron in one of its rows;
# B               - Vector of biases' vectors;
# F               - Vector of vectors of activation functions.

# If there is only a vector of functions for the activation functions, 
# i.e. all the activation functions in the same layer are the same

function evaluate_ANN(X::Vector{Float64}, W::Vector{Matrix{T}},
 B::Vector{Vector{T}}, F::Vector{Function}) where {T<:Number}

    # Iterates from the input to the output layers

    for i=1:length(W)

        # Multiplies the input by the weights matrix. Adds the biases
        # and evaluates the activation functions

        X = F[i].((W[i]*X) .+ B[i])

    end

    # Returns the ANN's output

    return X

end 

# If there is a vector of vectors of activation functions, i.e. they are
# individually and independently set

function evaluate_ANN(X::Vector{T}, W::Vector{Matrix{T2}}, B::Vector{
 Vector{T2}}, F::Vector{Vector{Function}}) where {T<:Number, T2<:Number}

    # Iterates from the input to the output layers

    for i=1:length(W)

        # Multiplies the input by the weights matrix. Adds the biases

        X = (W[i]*X) .+ B[i]

        # Evaluates the activation functions

        for j=1:length(F[i])

            X[j] = F[i][j](X[j])

        end

    end

    # Returns the ANN's output

    return X

end 

# Receiving the parameters in a single long vector

# Defines a function to evaluate the response of an ANN.
# The arguments are:
# X              - ANN's input vector;
# params_vector  - vector with biases and weights;
# neurons_number - vector with the number of neurons in each one of the
#                  layers
# F              - Vector of vectors of activation functions.

# If there is only a vector of functions for the activation functions, 
# i.e. all the activation functions in the same layer are the same

function evaluate_ANN(X::Vector{T}, params_vector::Vector{T2},
 neurons_number::Vector{Int64}, F::Vector{Function}) where {T<:Number, T2<:Number}

    # Initializes a counter of recovered elements

    read_elements = 0

    # Iterates through the layers

    for i=2:length(neurons_number)

        # Creates the result of the linear combination for each neuron
        # in the layer

        X_linearComb = Vector{Number}(undef, neurons_number[i])

        # Iterates through the number of neurons in the layer

        for j=1:neurons_number[i]

            X_linearComb[j] = (dot(params_vector[(read_elements+1):(
             read_elements+neurons_number[i-1])], X)+params_vector[(
             read_elements+neurons_number[i-1]+1)])

            # Updates the number of recovered elements

            read_elements += neurons_number[i-1]+1

        end

        # Evaluates the activation functions

        X = F[i-1].(X_linearComb)


    end

    # Returns the ANN's output


    return X


end 

# If there is a vector of vectors of activation functions, i.e. they are
# individually and independently set

function evaluate_ANN(X::Vector{T}, params_vector::Vector{T2},
 neurons_number::Vector{Int64}, F::Vector{Vector{Function}}) where {T<:Number, T2<:Number}

    # Initializes a counter of recovered elements

    read_elements = 0

    # Iterates through the layers

    for i=2:length(neurons_number)

        # Creates the result of the linear combination for each neuron
        # in the layer

        X_linearComb = Vector{Number}(undef, neurons_number[i])

        # Iterates through the number of neurons in the layer

        for j=1:neurons_number[i]

            X_linearComb[j] = F[i-1][j](dot(params_vector[(read_elements
             +1):(read_elements+neurons_number[i-1])], X)+params_vector[(
             read_elements+neurons_number[i-1]+1)])

            # Updates the number of recovered elements

            read_elements += neurons_number[i-1]+1

        end

    end

    # Returns the ANN's output

    return X

end 

########################################################################
#                            Initialization                            #
########################################################################

########################################################################
#                             Vector format                            #
########################################################################

# Defines a function to initialize the weights and biases of an ANN mo-
# del randomly, with a normal distribution centered in zero and with u-
# nitary standard deviation

function initialize_randomlyVector(neurons_number::Vector{Int64},
 distribution_radius::Float64)

    # Initializes a number of the total amount of parameters, i.e., the
    # sum of quantity of weights with the quantity of biases

    parameters_counter = 0

    # Iterates through the layers to update the vectors and matrices 
    # randomly

    for i=2:length(neurons_number)

        # Adds the number of weights and the number of biases

        parameters_counter += (neurons_number[i]*(neurons_number[i-1]+1))

    end

    # Returns a vector of parameters randomly created

    return (distribution_radius*randn(parameters_counter))

end

########################################################################
#                            Matrix format                             #
########################################################################

# Defines a function to initialize the weights and biases of an ANN mo-
# del randomly, with a normal distribution centered in zero and with u-
# nitary standard deviation

function initialize_randomlyMatrix(neurons_number::Vector{Int64},
 distribution_radius::Float64)

    # Recovers the total number of layers 

    n_layers = length(neurons_number)

    # Initializes the vector of biases

    B = Vector{Vector{Float64}}(undef, n_layers-1)

    # Initializes the vector of vectors of weights

    W = Vector{Matrix{Float64}}(undef, n_layers-1)

    # Iterates through the layers to update the vectors and matrices 
    # randomly

    for i=2:n_layers

        # Updates the vector of biases

        B[i-1] = (distribution_radius*randn(neurons_number[i]))

        # Updates the matrix of weights

        W[i-1] = (distribution_radius*randn(neurons_number[i], 
         neurons_number[i-1]))

    end

    # Returns the vectors

    return W, B

end