# Routine to store functions to perform tests

using LinearAlgebra

using ForwardDiff

using Plots

using TimerOutputs

#using Optimization, OptimizationNLopt

using Plots

using LaTeXStrings

using WallE

include("evaluation_and_initialization.jl")

include("activation_functions.jl")

include("utilities.jl")

include("loss_functions.jl")

include("sensitivity_analysis.jl")

include("loss_pinns.jl")

include("domain_residues.jl")

include("boundary_residues.jl")

include("augmented_lagrangian_methods.jl")

include("plot_utilities.jl")

include("ADAM.jl")

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

    distribution_radius = 5.0

    # Set a vector with the number of neurons in each one of the layers

    neurons_number = [2; 2; 1]

    # Sets the vector of activation functions considering that they are 
    # all the same in the same layer

    F = Vector{Function}(undef, length(neurons_number)-1)

    F[1] = linear_function
    
    F[2] = linear_function

    # Initializes the weights and biases randomly

    params_vector = initialize_randomlyVector(neurons_number,
     distribution_radius)

    #params_vector = [1.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.5; 0.5; 0.0]+(0.5*randn(9))

    # Defines the minimum value of the objetive function
 
    minimum_objective = 1E-5

    # Defines the number of iterations

    n_iterations = 10000
 
    # Defines the optimizer of the NLopt library
 
    optimizer = :LD_TNEWTON

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

    ####################################################################
    #                        Boundary conditions                       #
    ####################################################################

    # Creates a grid of boundary points for Dirichlet boundary condi-
    # tions. Creates a vector of the value of the boundary condition and
    # another vector with the indexes of the ouput layer to be constrained
    # with boundary condition

    dOmega_collocationDirichlet = Vector{Matrix{Float64}}(undef, 2)

    dOmega_valuesDirichlet = Vector{Matrix{Float64}}(undef, 2)

    dOmega_outputIndexesDirichlet = Vector{Matrix{Int64}}(undef, 2)

    dOmega_collocationDirichlet[1] = [0.0 0.2 0.5 0.8 1.0;
                                   0.0 0.0 0.0 0.0 0.0]

    dOmega_valuesDirichlet[1] = [0.0 (0.5*(0.2^2)) (0.5*(0.5^2)) (0.5*(0.8^2)) 0.5]

    dOmega_outputIndexesDirichlet[1] = [1 1 1 1 1]

    dOmega_collocationDirichlet[2] = [0.0 0.0 0.0 0.0 0.0;
                                   0.0 0.2 0.5 0.8 1.0]

    dOmega_valuesDirichlet[2] = [0.0 (0.5*(0.2^2)) (0.5*(0.5^2)) (0.5*(0.8^2)) 0.5]

    dOmega_outputIndexesDirichlet[2] = [1 1 1 1 1]

    # Creates a grid of boundary points for Neumann boundary conditions.
    # Creates a vector of sets of Neumann collocation points in the 
    # boundary; a vector of matrices of values of the Neumann conditions
    # and a vector of functions, each of these functions picks the deri-
    # vatives of the ANN and performs the boundary condition

    dOmega_collocationNeumann = Matrix{Float64}[]

    dOmega_valuesNeumann = Matrix{Float64}[]

    Neumann_conditions = Function[]

    # Calculates the number of collocation points of the domain

    n_collocationPoints = size(omega_collocationPoints,2)

    # Calculates the number of boundary points

    n_boundaryPoints = 10

    # Defines a metric for the residue

    function domain_residueMetric(r::Vector{T}) where {T<:Number}
    
        return sum(r.^2)/n_collocationPoints
end

    # Defines a metric for the error in the boundary

    function boundary_residueMetric(r::Vector{T}) where {T<:Number}
    
        return sum(r.^2)/n_boundaryPoints
end

    ####################################################################
    #                              Drivers                             #
    ####################################################################

    # Creates a driver for the loss function in the domain

    function driver_lossDomain(parameters::Vector{T}) where {T<:Number}

        # Creates a driver for the model's output setting the parameters

        driver_input(input) = evaluate_ANN(input, parameters,
         neurons_number, F)

        # Evaluates the parcels of the loss function in the domain and
        # returns it

        return phi_lossDomain(driver_input, parameters,
         omega_collocationPoints, residue_domain, domain_residueMetric)

    end

    # Creates a driver for the value of the error in the boundary points

    function driver_constraintsBoundary(parameters::Vector{T}) where {T<:Number}

        # Creates a driver for the model's output setting the parameters

        driver_input(input) = evaluate_ANN(input, parameters,
         neurons_number, F)

        # Evaluates the parcels of the loss function in the boundary and
        # returns it

        return phi_lossBoundary(driver_input, parameters,
         dOmega_collocationDirichlet, dOmega_valuesDirichlet,
         dOmega_outputIndexesDirichlet, dOmega_collocationNeumann,
         dOmega_valuesNeumann, Neumann_conditions, dirichlet_error, 
         neumann_error, boundary_residueMetric)

    end

    # Creates a driver for the total loss by just summing all parcels

    function driver_totalLoss(parameters::Vector{T}) where {T<:Number}

        return (driver_lossDomain(parameters)+sum(
         driver_constraintsBoundary(parameters)))

    end

    # Creates a driver for the gradient 

    function driver_gradient(parameters::Vector{T}) where {T<:Number}

        # Evaluates the gradient and returns it
    
        return gradient_automaticDiff(driver_totalLoss, parameters)

    end

    ####################################################################
    #                            Calculation                           #
    ####################################################################

    # Evaluates the loss function in the domain

    loss_domain = driver_lossDomain(params_vector)

    println("The loss of the domain collocation points with the initia",
     "l vector of parameters is ", loss_domain, "\n\n")

    # Evaluates the loss function in the boundary

    loss_boundary = driver_constraintsBoundary(params_vector)

    println("The vector of error of the boundary collocation points wi",
     "th the initial vector of parameters is:")

    for i=1:length(loss_boundary)

        println(loss_boundary[i])

    end

    # Evaluates the total loss

    total_loss = driver_totalLoss(params_vector)

    println("\n\nThe total loss with the initial vector of parameters ",
     "is ", total_loss, "\n")

    # Makes a surface plot

    function driver_input(input)
    evaluate_ANN(input, params_vector, 
     neurons_number, F)
end

    plot_surface([0.0; 1.0], [0.0; 1.0], driver_input, path_plot=
     joinpath(pwd(), "prior_optimization.pdf"))

    ####################################################################
    #                           Optimization                           #
    ####################################################################

    # Tests the gradient

    println("The gradient of the objective function with the initial v",
     "ector of parameters and the numerical gradient are:")

    gradient_vector = driver_gradient(params_vector)

    numerical_gradient = gradient_CFD(driver_totalLoss, params_vector, 
     epsilon=5E-8)

    for i=1:length(params_vector)

        println(gradient_vector[i], "  ", numerical_gradient[i], "  ",
         params_vector[i])

    end 

    println("\n\n")

    # Sets and uses the WallE optimizer

    #=ci = -20*ones(length(params_vector))

    cs = 20*ones(length(params_vector))

    options = WallE.Init()

    options["NITER"] = n_iterations

    options["TOL_NORM"] = 1E-5

    options["SHOW"] = true 

    output = WallE.Solve(driver_totalLoss, driver_gradient, 
     params_vector, ci, cs, options)

    x = output["RESULT"]=#

    x_adam = adam_optimizer(params_vector, driver_totalLoss,
     driver_gradient, length(params_vector))

    #= Initializes the NLopt optimization problem

    opt = Opt(optimizer, length(params_vector))

    opt.maxeval = n_iterations

    opt.ftol_abs = minimum_objective
    
    opt.min_objective = driver_optimization
    
    min_loss, params_vector, convergence_flag = optimize(opt,
     params_vector)=#

end

# Defines a function to test Augmented Lagrangian method

function test_pinnLA()

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

    ####################################################################
    #                       Augmented Lagrangian                       #
    ####################################################################

    # Defines the number of outer iterations

    n_iterationsLA = 10

    # Defines the maximum number of inner iterations

    n_innerIterations = 10000

    # Defines the c penalty

    c_penalty = 100.0

    # Defines the minimum value of the objetive function

    minimum_objective = 1E-5

    # Defines the optimizer of the NLopt library

    optimizer = :LD_LBFGS_NOCEDAL

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

    # Creates a grid of boundary points for Dirichlet boundary condi-
    # tions

    dOmega_collocationDirichlet = Vector{Matrix{Float64}}(undef, 2)

    dOmega_collocationDirichlet[1] = [0.0 0.2 0.5 0.8 1.0;
                                   0.0 0.0 0.0 0.0 0.0;
                                   0.0 (0.5*(0.2^2)) (0.5*(0.5^2)) (0.5*(0.8^2)) 0.5]

    dOmega_collocationDirichlet[2] = [0.0 0.0 0.0 0.0 0.0;
                                   0.0 0.2 0.5 0.8 1.0;
                                   0.0 (0.5*(0.2^2)) (0.5*(0.5^2)) (0.5*(0.8^2)) 0.5]

    # Creates a grid of boundary points for Neumann boundary conditions

    dOmega_collocationNeumann = Matrix{Float64}[]

    # Calculates the number of collocation points

    n_collocationPoints = size(omega_collocationPoints,2)

    # Calculates the number of boundary points

    n_boundaryPoints = 10

    # Defines a metric for the residue

    domain_residueMetric(r) = sum(r.^2)/n_collocationPoints

    # Defines a metric for the error in the boundary

    boundary_residueMetric(r) = sum(r.^2)/n_boundaryPoints

    # Sets the vector of Lagrange multipliers

    lagrange_multipliers = [1.0; 1.0]

    # Creates a driver for the loss function in the domain

    function driver_lossDomain(parameters::Vector{T}) where {T<:Number}

        # Creates a driver for the model's output setting the parameters

        driver_input(input) = evaluate_ANN(input, parameters,
         neurons_number, F)

        # Evaluates the parcels of the loss function in the domain and
        # returns it

        return phi_lossDomain(driver_input, parameters,
         omega_collocationPoints, residue_domain, domain_residueMetric, 
         neurons_number[1], neurons_number[end])

    end

    # Creates a driver for the value of the error in the boundary points

    function driver_constraintsBoundary(parameters::Vector{T}) where {T<:Number}

        # Creates a driver for the model's output setting the parameters

        driver_input(input) = evaluate_ANN(input, parameters,
         neurons_number, F)

        # Evaluates the parcels of the loss function in the boundary and
        # returns it

        return phi_lossBoundary(driver_input, parameters,
         dOmega_collocationDirichlet, dOmega_collocationNeumann,
         dirichlet_error, neumann_error, boundary_residueMetric,
         neurons_number[1], neurons_number[end], lagrange_multipliers)[2]

    end

    # Creates a driver for the evaluation of the gradient of the domain

    function driver_gradientDomain(parameters::Vector{Float64})

        # Evaluates the gradient and returns it

        return gradient_automaticDiff(driver_lossDomain, parameters)

    end

    # Creates a driver for the evaluation of the gradient of the boundary

    function driver_gradientConstraints(parameters::Vector{Float64})

        # Evaluates the jacobian and returns it

        return jacobian_automaticDiff(driver_constraintsBoundary, parameters)

    end

    # Evaluates the loss function 

    phi_PINNInitial = (driver_lossDomain(params_vector)+dot(
     driver_constraintsBoundary(params_vector), lagrange_multipliers))

    # Tests the gradient

    driver_gradientDomain(params_vector)

    driver_gradientConstraints(params_vector)

    # Evaluates the augmented lagrangian

    augmented_LagrangianMartinez(params_vector, driver_lossDomain, 
     driver_constraintsBoundary, driver_gradientDomain, 
     driver_gradientConstraints, length(dOmega_collocationDirichlet)+
     length(dOmega_collocationNeumann), c_penalty, n_iterationsLA, 
     n_innerIterations, optimizer, minimum_objective)

end

test_pinn()