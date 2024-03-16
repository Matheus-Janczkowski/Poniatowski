# Routine to store functions to perform tests

using LinearAlgebra

using ForwardDiff

using Plots

using TimerOutputs

using NLopt

include("evaluation_and_initialization.jl")

include("activation_functions.jl")

include("utilities.jl")

include("loss_functions.jl")

include("sensitivity_analysis.jl")

include("loss_pinns.jl")

include("augmemted_lagrangian_methods.jl")

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

    neurons_number = [1; 2; 1]

    # Initializes the input

    input = [1.0]#randn(neurons_number[1])

    # Initializes a random expected output

    output_true = randn(neurons_number[end])

    # Sets the vector of activation functions considering that they are 
    # all the same in the same layer
    
    # TODO: corrigir

    F = Vector{Vector{Function}}(undef, length(neurons_number)-1)

    F[1] = [quadratic; quadratic]
    
    F[2] = [linear]

    F = [quadratic; linear]

    # Initializes the weights and biases randomly

    params_vector = initialize_randomlyVector(neurons_number,
     distribution_radius)

    params_vector = [1.0; 1.0; 1.0; 1.0; 2.0; 2.0; 2.0; 2.0]

    # Defines a driver for the loss function

    function driver_loss(params_vector)

        # Evaluates the ANN output
    
        output = evaluate_ANN(input, params_vector, neurons_number, F)

        # Calculates the loss function and returns it

        return loss_function(output.-output_true)

    end

    # Defines a driver for the output of the ANN model as function of 
    # the inputs only

    function driver_model(input)

        # Evaluates the ANN output and returns it

        return evaluate_ANN(input, params_vector, neurons_number, F)

    end

    @show driver_model(input)

    # Evaluates the gradient of the loss function

    gradient_params = gradient_automaticDiff(driver_loss, params_vector)

    println("The gradient of the loss function is ", gradient_params,
     "\n")

    # Evaluates the jacobian of the response of the ANN model w.r.t. the
    # input

    jacobian_input = jacobian_automaticDiff(driver_model, input)

    println("The jacobian of the ANN model w.r.t. the input is ", 
     jacobian_input, "\n")

    # Evaluates the hessian of the response of the ANN model w.r.t. the
    # input

    hessian_input = hessian_automaticDiff(driver_model, input, neurons_number[end])

    println("The hessian of the ANN model w.r.t. the input is ",
     hessian_input, "\n")

    return driver_loss, params_vector

end

# Defines a function to test PINNs

function test_pinn()

    ####################################################################
    #                                ANN                               #
    ####################################################################
    
    # Sets a radius for the distribution of initial weights and biases

    distribution_radius = 100.0

    # Set a vector with the number of neurons in each one of the layers

    neurons_number = [2; 2; 1]

    # Sets the vector of activation functions considering that they are 
    # all the same in the same layer

    F = [quadratic; linear]

    # Initializes the weights and biases randomly

    params_vector = initialize_randomlyVector(neurons_number,
     distribution_radius)

    # Defines a driver for the output of the ANN model as function of 
    # the inputs and of the parameters

    function pinn_model(input, parameters)

        # Evaluates the ANN output and returns it

        return evaluate_ANN(input, parameters, neurons_number, F)

    end

    ####################################################################
    #                                PDE                               #
    ####################################################################

    # Sets the vector of number of collocation points in the x and y di-
    # rections

    grid_discretization = [8; 5]

    # Sets the interval of the domain (x_inf <= x <= x_sup and y_inf <= 
    # y <= y_sup)

    interval_dimensions = [0.0 1.0; 0.0 0.5]


    # Creates a grid of domain collocation points

    omega_collocationPoints = [0.2 0.5 0.8 0.2 0.5 0.8 0.2 0.5 0.8;
                               0.2 0.2 0.2 0.5 0.5 0.5 0.8 0.8 0.8]

    # Creates a grid of boundary points

    dOmega_collocationPoints = Vector{Matrix{Float64}}(undef, 2)

    dOmega_collocationPoints[1] = [0.0 0.2 0.5 0.8 1.0;
                                   0.0 0.0 0.0 0.0 0.0;
                                   0.0 (0.5*(0.2^2)) (0.5*(0.5^2)) (0.5*(0.8^2)) 0.5]

    dOmega_collocationPoints[2] = [0.0 0.0 0.0 0.0 0.0;
                                   0.0 0.2 0.5 0.8 1.0;
                                   0.0 (0.5*(0.2^2)) (0.5*(0.5^2)) (0.5*(0.8^2)) 0.5]

    # Calculates the number of collocation points

    n_collocationPoints = size(omega_collocationPoints,2)

    # Calculates the number of boundary points

    n_boundaryPoints = 10

    # Defines a metric for the residue

    domain_residueMetric(r) = sum(r.^2)/n_collocationPoints

    # Defines a metric for the error in the boundary

    boundary_residueMetric(r) = sum(r.^2)/n_boundaryPoints

    # Sets the vector of Lagrange multipliers

    lagrange_multipliers = [1.0; 1.0; 1.0]

    # PDE: du/dx + du/dy = x+y and u(x,0) = 0.5*x^2 and u(0,y) = 0.5*y^2
    # Defines a function for the residue of the domain points (individu-
    # ally)

    function residue_domain(input::Vector{Float64}, pinn_model::Function)

        # Evaluates the derivatives

        jacobian_input = jacobian_automaticDiff(pinn_model, input)

        # Evaluates the residue and returns it

        return (jacobian_input[1,1]+jacobian_input[1,2]-input[1]-input[2])

    end

    # Defines a function for the residue of the boundary points (indivi-
    # dually)

    function dirichlet_error(input::Vector{Float64}, pinn_model::Function, 
     true_value::Vector{Float64})

        return (pinn_model(input)-true_value)

    end

    # Evaluates the loss function 

    phi_PINNInitial = phi_lossPINN(params_vector, pinn_model,
     omega_collocationPoints, dOmega_collocationPoints, residue_domain,
     dirichlet_error, domain_residueMetric, boundary_residueMetric,
     neurons_number[1], neurons_number[end], lagrange_multipliers)

    # Creates a driver for the loss function

    driver_loss(parameters) = phi_lossPINN(parameters, pinn_model,
    omega_collocationPoints, dOmega_collocationPoints, residue_domain,
    dirichlet_error, domain_residueMetric, boundary_residueMetric,
    neurons_number[1], neurons_number[end], lagrange_multipliers)[1]

    # Creates a driver for the gradient of the loss function w.r.t. the
    # parameters

    driver_gradient(parameters) = gradient_automaticDiff(driver_loss, 
     parameters)

    # Tests the gradient

    driver_gradient(params_vector)

end