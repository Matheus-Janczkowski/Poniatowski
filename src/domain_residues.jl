# Routine to store the residues of different PDEs

# PDE: du/dx + du/dy = x+y and u(x,0) = 0.5*x^2 and u(0,y) = 0.5*y^2
# Defines a function for the residue of the domain points (individu-
# ally)

function residue_domain(input::Vector{Float64}, pinn_model::Function)

    # Evaluates the derivatives

    jacobian_input = jacobian_automaticDiff(pinn_model, input)

    # Evaluates the residue and returns it

    return ([jacobian_input[1,1]+jacobian_input[1,2]-input[1]-input[2]])

end