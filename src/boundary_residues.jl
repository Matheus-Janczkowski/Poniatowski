# Routine to store errors and residues in the boundary

# Defines a function for the residue of the boundary points (individual-
# ly) with Dirichlet boundary conditions

function dirichlet_error(input::Vector{Float64}, pinn_model::Function, 
 true_value::Vector{Float64}, output_neurons::Vector{Int64})

    return (pinn_model(input)[output_neurons]-true_value)

end

# Defines a function for the residue of the boundary points (individual-
# ly) with Neumann boundary conditions

function neumann_error(input::Vector{Float64}, pinn_model::Function,
 condition_operator::Function, true_value::Vector{Float64})

    # Evaluates the derivatives

    jacobian_input = jacobian_automaticDiff(pinn_model, input)

    # Makes some operation with the derivatives

    return (condition_operator(jacobian_input, input)-true_value)

end