# Routine to store some utility functions for the problem of ANNs

# Defines a function to convert a vector of parameters to a matrix of 
# weights and biases

function convert_vectorMatrices(params_vector::Vector{Float64},
 neurons_number::Vector{Int64})

    # Recovers the total number of layers 

    n_layers = length(neurons_number)

    # Initializes the vector of biases

    B = Vector{Vector{Float64}}(undef, n_layers-1)

    # Initializes the vector of vectors of weights

    W = Vector{Matrix{Float64}}(undef, n_layers-1)

    # Initializes a counter of elements that have been recovered

    read_elements = 0

    # Iterates through the layers to update the vectors and matrices 
    # randomly

    for i=2:n_layers

        # Initializes the vector of biases and the matrix of weights of
        # this layer

        B[i-1] = Vector{Float64}(undef, neurons_number[i])

        W[i-1] = Matrix{Float64}(undef, neurons_number[i], 
         neurons_number[i-1])

        # Recovers the elements of the weights and biases

        for j=1:neurons_number[i]

            W[i-1][j,1:neurons_number[i-1]] = params_vector[(
             read_elements+1):(read_elements+neurons_number[i-1])]

            # Updates the count of receovered elements

            read_elements += neurons_number[i-1]+1

            B[i-1][j] = params_vector[read_elements]

        end

    end

    # Returns the vectors

    return W, B

end

# Defines a function to convert the matrices to vectors 

function convert_matricesVector(W::Vector{Matrix{Float64}}, b::Vector{ 
 Vector{Float64}}, neurons_number::Vector{Int64})

    # Recovers the total number of layers 

    n_layers = length(neurons_number)

    # Initializes the vector of parameters

    params_vector = Float64[]

    # Iterates through the layers to update the vectors and matrices 
    # randomly

    for i=2:n_layers

        # Recovers and adds the elements of the weights and biases vec-
        # tors to the vector of parameters

        for j=1:neurons_number[i]

            params_vector = [params_vector; W[i-1][j,:]; b[i-1][j]]

        end

    end

    # Returns the vector of parameters

    return params_vector

end