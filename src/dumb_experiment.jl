# Routine to do a dumb test 

using LinearAlgebra

using DelimitedFiles

using ForwardDiff

using NLopt

using WallE

using Plots 

using LaTeXStrings

include("plot_utilities.jl")

# Defines a function for the following PINN model
#
# x -- x21 \
#   /\      x31
# y -- x22 /

quadratic(x) = x^2

linear(x) = x

function pinn_model(x, params_vector, neurons_number, 
 activation_functions)

    #=x_21 = (((x[1]*params_vector[1])+(x[2]*params_vector[2])+
     params_vector[3])^2)

    x_22 = (((x[1]*params_vector[4])+(x[2]*params_vector[5])+
     params_vector[6])^2)

    x_31 = ((x_21*params_vector[7])+(x_22*params_vector[8])+
     params_vector[9])

    return x_31=#

    read_elements = 0

    for i=2:length(neurons_number)

        X_linearComb = []

        for j=1:neurons_number[i]

            push!(X_linearComb, (dot(x, params_vector[(read_elements+1):(
             read_elements+neurons_number[i-1])])+params_vector[(
             read_elements+neurons_number[i-1]+1)]))

            read_elements += neurons_number[i-1]+1

        end

        x = activation_functions[i-1].(X_linearComb)

    end

    return x

end

# Defines a function for the analytic derivative of the PINN model 
# w.r.t. the input

function analytic_derivative(params_vector, input)

    w_1121, w_1221, b_21, w_1122, w_1222, b_22, w_2131, w_2231, b_31 = params_vector

    df21_dz = 2*((w_1121*input[1])+(w_1221*input[2])+b_21)

    df22_dz = 2*((w_1122*input[1])+(w_1222*input[2])+b_22) 

    df31_dz = 1.0

    du_dx = (df31_dz*((w_2131*w_1121*df21_dz)+(w_2231*w_1122*df22_dz)))

    du_dy = (df31_dz*((w_2131*w_1221*df21_dz)+(w_2231*w_1222*df22_dz)))

    return du_dx, du_dy
    
end

# Defines a function for the residue in the domain

function residue_domain(params_vector::Vector{T}) where {T<:Number}

    neurons_number = [2; 2; 1]

    activation_functions = [quadratic; linear]

    omega_collocationPoints = [0.2 0.5 0.8 0.2 0.5 0.8 0.2 0.5 0.8;
                               0.2 0.2 0.2 0.5 0.5 0.5 0.8 0.8 0.8]

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

    pinn_domain(x) = pinn_model(x, params_vector, neurons_number,
     activation_functions)

    phi = zero(T)

    for i=1:size(omega_collocationPoints,2)

        derivatives = ForwardDiff.jacobian(pinn_domain, 
         omega_collocationPoints[:,i])

        #println("AutoDiff: du_dx = ", derivatives[1,1], " du_dy = ", derivatives[2,1], "\n")

        phi += (sum(derivatives)-sum(omega_collocationPoints[:,i]))^2

    end

    for i=1:length(dOmega_collocationDirichlet)

        for j=1:size(dOmega_collocationDirichlet[i],2)

            phi += sum((pinn_model(dOmega_collocationDirichlet[i][:,j],
            params_vector).-dOmega_valuesDirichlet[i][:,j]).^2)

        end

    end

    return phi

end

function residue_domainAnalytic(params_vector::Vector{T}) where {T<:Number}

    neurons_number = [2; 2; 1]

    activation_functions = [quadratic; linear]

    omega_collocationPoints = [0.2 0.5 0.8 0.2 0.5 0.8 0.2 0.5 0.8;
                               0.2 0.2 0.2 0.5 0.5 0.5 0.8 0.8 0.8]

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
                                
    phi = zero(T)

    for i=1:size(omega_collocationPoints,2)

        du_dx, du_dy = analytic_derivative(params_vector, omega_collocationPoints[:,i])

        #println("Analytic: du_dx = ", du_dx, " du_dy = ", du_dy, "\n")

        phi += (du_dx+du_dy-sum(omega_collocationPoints[:,i]))^2

    end

    for i=1:length(dOmega_collocationDirichlet)

        for j=1:size(dOmega_collocationDirichlet[i],2)

            phi += sum((pinn_model(dOmega_collocationDirichlet[i][:,j],
             params_vector, neurons_number, activation_functions).-
             dOmega_valuesDirichlet[i][:,j]).^2)

        end

    end

    return phi

end

# Defines a function to test the gradient 

function test_gradient()

    params_vector = [1.0; 2.0; -1.0; 10.0; -20.0; 15.0; 2.0; -1.0; 6.0]

    gradient_vector = ForwardDiff.gradient(residue_domain, params_vector)

    gradient_vectorAnalytic = ForwardDiff.gradient(residue_domainAnalytic,
     params_vector)

    for i=1:length(params_vector)

        println(gradient_vector[i], "  ", gradient_vectorAnalytic[i])

    end

end

function test_crossedDerivative()

    x = randn(2)

    f(x) = (x[1]^2)*(x[2]^3)

    df_dx(x) = ForwardDiff.gradient(f, x)[1]

    ddf_dxdy = ForwardDiff.gradient(df_dx, x)[2]

    analytic = 6*x[1]*(x[2]^2)

    println("Analytic = ", analytic, " AutoDiff = ", ddf_dxdy)
    
end

function test_crossedFunction()

    x = randn(2)

    z = randn(2)

    function g(z)

        f(x) = ((x[1]^2)*(x[2]^3)*(z[1]^4))+(z[2]^2)

        df_dx = ForwardDiff.gradient(f, x)

        return sum(df_dx)
        
    end

    ddf_dxdy = ForwardDiff.gradient(g, z)

    analytic = [4*((2*x[1]*(x[2]^3))+(3*(x[1]^2)*(x[2]^2)))*(z[1]^3); 0.0]

    println("Analytic = ", analytic, " AutoDiff = ", ddf_dxdy)
    
end

# Defines a function to test optimization

function test_optimization()

    gradient_objective(params) = ForwardDiff.gradient(residue_domain, params)

    gradient_analytic(params) = ForwardDiff.gradient(residue_domainAnalytic,
    params_vector)

    params_vector = randn(9)

    println(norm(gradient_analytic(params_vector)-gradient_objective(
     params_vector)))

    initial_model(input) = pinn_model(input, params_vector)
    
    plot_surface([0.0; 1.0], [0.0; 1.0], initial_model, path_plot=
    joinpath(pwd(), "after_optimization.pdf"), title=string("u(x,y),\\;"*
     "\\varphi="*string(residue_domain(params_vector))))

    ci = -20*ones(length(params_vector))

    cs = 20*ones(length(params_vector))

    options = WallE.Init()

    options["NITER"] = 1000

    options["TOL_NORM"] = 1E-5

    options["SHOW"] = true 

    output = WallE.Solve(residue_domain, gradient_objective,
     params_vector, ci, cs, options)

    x = output["RESULT"]

    final_model(input) = pinn_model(input, x)
    
    plot_surface([0.0; 1.0], [0.0; 1.0], final_model, path_plot=
    joinpath(pwd(), "after_optimization.pdf"), title=string("u(x,y),\\;"*
    "\\varphi="*string(residue_domain(x))))

    writedlm(joinpath(pwd(), "trained_parameters.txt"), x)

    return x
    
end