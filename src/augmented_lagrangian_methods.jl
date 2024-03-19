# Routine to store functions for the augmented lagrangian methods

# Defines a function for the augmented lagrangian of Martinez

function augmented_LagrangianMartinez(params_vector::Vector{Float64},
 driver_lossDomain::Function, driver_constraintsBoundary::Function,
 driver_gradientDomain::Function, driver_gradientConstraints::Function,
 n_constraints::Int64, c_penalty::Float64, n_iterationsLA::Int64, 
 n_innerIterations::Int64, optimizer::Symbol, minimum_objective::Float64)

    # Initializes the vector of lagrange_multipliers with zeros

    lagrange_multipliers = zeros(n_constraints)

    # Initializes a history of the loss function

    history_loss = [driver_lossDomain(params_vector)]

    # Initializes a history of the norm of the difference of the lagrange
    # multipliers from one iteration to the next

    history_lagrange = Float64[]

    # Initializes the NLopt optimization problem

    opt = Opt(:LD_MMA, length(params_vector))

    opt.maxeval = n_innerIterations

    opt.ftol_abs = minimum_objective

    # Iterates through the number of iterations

    for j=1:n_iterationsLA

        # The following driver is updated since the lagrange multipliers
        # are updated at each iteration of the Augmented Lagrangian me-
        # thod

        function driver_optimization(parameters::Vector{Float64}, dLA::
         Vector{Float64})

            # Initializes the value

            LA = 0.0

            # Evaluates the constraints

            constraints_vector = driver_constraintsBoundary(parameters)

            # Calculates the gradient of the constraints

            gradient_constraints = driver_gradientConstraints(parameters)

            # Iterates through the constraints

            for i=1:n_constraints

                # Sums the constraint

                LA += (positive_operator((lagrange_multipliers[i]/
                 c_penalty)+constraints_vector[i])^2)

                dLA = (dLA+(positive_operator((lagrange_multipliers[i]/
                 c_penalty)+constraints_vector[i])*gradient_constraints[i,:]))

            end

            # Multiplies the gradient vector by c and adds the parcel 
            # relative to the loss in the domain

            dLA = ((c_penalty*dLA)+driver_gradientDomain(parameters))

            # Multiplies the loss function by (c/2) and adds the parcel
            # of the domain, then, returns it

            return ((0.5*c_penalty*LA)+driver_lossDomain(parameters))

        end

        function teste(x, dx)

            dx = 2*x 

            return dot(x,x)

        end

        # Updates the function to be minimized

        opt.min_objective = driver_optimization

        # Finds the minimum

        min_loss, params_vector, convergence_flag = optimize(opt,
         params_vector)

        # Updates the history

        push!(history_loss, min_loss)

        # Updates the lagrangian multipliers

        constraints_vector = driver_constraintsBoundary(params_vector)

        delta_lagrange = positive_operator.(lagrange_multipliers+(
         c_penalty*constraints_vector))

        push!(history_lagrange, norm(delta_lagrange))

        lagrange_multipliers .+= delta_lagrange

    end

    # Returns the minimum solution and the lagrangian multipliers

    return (params_vector, lagrange_multipliers, history_loss,
     history_lagrange)

end

# Defines a function for the positive only operator

function positive_operator(x)

    return max(0.0, x)

end