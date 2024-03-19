# Routine to store the optimization algorithms

########################################################################
#                              Optimizers                              #
########################################################################

# Defines a function for the ADAM optimizer

function adam_optimizer(x, objective_function, grad_function,
 dimensionality; stop_tol=1E-7, n_iterations=100000, alpha=1E-3,
 beta1=0.9, beta2=0.999, epsilon=1E-8, flag_plot=false)

    # Initializes a history of the points
    
    historico_plot = copy(x)
   
    # Initializes m, v and t

    t = 0

    m_t = zeros(dimensionality)

    v_t = zeros(dimensionality)

    # Iterates through the given number of iterations

    for i=1:n_iterations

        # Updates t

        t += 1

        # Calculates the gradient

        g_t = grad_function(x)

        # Tests the norm of the gradient as stopping condition

        norm_grad = norm(g_t)

        if norm_grad<stop_tol

            println("The optimization procedure was stopped at iterati",
            "on ", i, ", with the norm of the gradient being ",
            norm_grad, "\n")

            break

        end

        # Calculates the moments, but allocates them in a vector that is
        # equal to the division between scaled m and the square root of
        # scaled v

        m_divV = Float64[]

        m_t = ((beta1*m_t)+((1-beta1)*g_t))

        for j=1:dimensionality

            v_t[j] = ((beta2*v_t[j])+((1-beta2)*(g_t[j]^2)))

            # Calculates the quotient

            push!(m_divV, (((1/(1-(beta1^t)))*m_t[j])/(sqrt((1/(1-(beta2
            ^t)))*v_t[j])+epsilon)))

        end

        # Updates the vector of design variables

        x = (x-(alpha*m_divV))

        # Updates the historico

        if flag_plot

            historico_plot = [historico_plot copy(x)]

        end

    end

    # Returns the last point

    if flag_plot

        return x, historico_plot

    else

        return x

    end
   
end