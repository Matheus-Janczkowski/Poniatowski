# Routine to store errors and residues in the boundary

# Defines a function for the residue of the boundary points (indivi-
# dually) with Dirichlet boundary conditions

function dirichlet_error(input::Vector{Float64}, pinn_model::Function, 
    true_value::Vector{Float64})

        return (pinn_model(input)-true_value)

    end

    # Defines a function for the residue of the boundary points (indivi-
    # dually) with Neumann boundary conditions

    function neumann_error(input::Vector{Float64}, pinn_model::Function,
    true_value::Vector{Float64})

        # Evaluates the derivatives

        jacobian_input = jacobian_automaticDiff(pinn_model, input)

        # Makes some operation with the derivatives

        return (jacobian_input[1,1]-true_value)

    end