# Routine to store functions to perform tests

using LinearAlgebra

using Zygote

using Plots

using TimerOutputs

using NLopt

include("evaluation_and_initialization.jl")

include("activation_functions.jl")

include("utilities.jl")

include("loss_functions.jl")

include("sensitivity_analysis.jl")

# Defines a function to test the functionality of the random initializa-
# tion of parameters and of the evaluation of the ANN model

function test_evaluationANN()

    # Sets a radius for the distribution of initial weights and biases

    distribution_radius = 100.0

    # Set a vector with the number of neurons in each one of the layers

    neurons_number = [2; 4; 1]

    # Initializes the input

    input = randn(neurons_number[1])

    # Sets the vector of activation functions considering that they are 
    # all the same in the same layer

    F = [Function[linear; linear; linear; linear]; linear]

    ####################################################################
    #          Experiments with matrix form for the parameters         #
    ####################################################################

    # Initializes the weights and biases randomly

    W, B = initialize_randomlyMatrix(neurons_number, distribution_radius)

    # Evaluates the ANN output

    output_paramsMatrix =  evaluate_ANN(input, W, B, F)

    # Tests with the vector form

    params_vector = convert_matricesVector(W, B, neurons_number)

    output_paramsVector = evaluate_ANN(input, params_vector,
     neurons_number, F)
     
    println("From matrix to vector:\nThe difference between the evalua",
     "tion using matrix and vector formulations are ", norm(
     output_paramsMatrix-output_paramsVector), "\n\n")

    ####################################################################
    #          Experiments with vector form for the parameters         #
    ####################################################################

    # Initializes the weights and biases randomly

    params_vector = initialize_randomlyVector(neurons_number,
     distribution_radius)

    # Evaluates the ANN output

    output_paramsVector = evaluate_ANN(input, params_vector,
    neurons_number, F)

    # Tests with the matrix form

    W, B = convert_vectorMatrices(params_vector, neurons_number)

    output_paramsMatrix =  evaluate_ANN(input, W, B, F)
     
    println("From vector to matrix:\nThe difference between the evalua",
     "tion using matrix and vector formulations are ", norm(
     output_paramsMatrix-output_paramsVector), "\n\n")

end

# Defines a function to test the loss functions

function test_lossFunctions(loss_function)

    # Sets a radius for the distribution of initial weights and biases

    distribution_radius = 100.0

    # Set a vector with the number of neurons in each one of the layers

    neurons_number = [2; 4; 1]

    # Initializes the input

    input = randn(neurons_number[1])

    # Initializes a random expected output

    output_true = randn(neurons_number[end])

    # Sets the vector of activation functions considering that they are 
    # all the same in the same layer

    F = [Function[linear; linear; linear; linear]; linear]

    # Initializes the weights and biases randomly

    params_vector = initialize_randomlyVector(neurons_number,
     distribution_radius)

    # Evaluates the ANN output

    output = evaluate_ANN(input, params_vector, neurons_number, F)

    # Tests the loss function with the difference between the true out-
    # put and the ANN output

    println("The value of the loss function is ", loss_function(output.-
     output_true), "\n")

end

# Defines a function to test the evaluation of the gradient of a loss
# function using automatic differentiation

function test_gradientAutoDiff(loss_function)
    
    # Sets a radius for the distribution of initial weights and biases

    distribution_radius = 100.0

    # Set a vector with the number of neurons in each one of the layers

    neurons_number = [2; 4; 1]

    # Initializes the input

    input = randn(neurons_number[1])

    # Initializes a random expected output

    output_true = randn(neurons_number[end])

    # Sets the vector of activation functions considering that they are 
    # all the same in the same layer

    F = [Function[linear; linear; linear; linear]; linear]

    # Initializes the weights and biases randomly

    params_vector = initialize_randomlyVector(neurons_number,
     distribution_radius)

    # Defines a driver for the loss function

    function driver_loss(params_vector)

        # Evaluates the ANN output
    
        output = evaluate_ANN(input, params_vector, neurons_number, F)

        # Calculates the loss function and returns it

        return loss_function(output.-output_true)

    end

    # Evaluates the gradient of the loss function

    gradient_vector = gradient_automaticDiff(driver_loss, params_vector)

    println("The gradient of the loss function is ", gradient_vector,
     "\n")

end