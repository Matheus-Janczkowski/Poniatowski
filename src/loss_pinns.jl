# Routine to store functions for the loss function of PINNs

# Defines a function to calculate the loss function of the PINN model

function phi_lossPINN(parameters::Vector{T}, pinn_model::Function, 
 omega_collocationPoints::Matrix{Float64}, dOmega_collocationPoints::
 Vector{Matrix{Float64}}, residue_domain::Function, dirichlet_error::
 Function, domain_residueMetric::Function, boundary_residueMetric::Function,
 input_dimensionality::Int64, output_dimensionality::Int64,
 lagrange_multipliers::Vector{Float64}) where {T<:Number}

    # Creates a driver for the model's output setting the parameters

    driver_input(input) = pinn_model(input, parameters)

    # Initializes the loss function for each one of the parcels
    
    phi_domain = zero(T)

    phi_boundary = Vector{T}(undef, length(dOmega_collocationPoints))

    # Iterates through the domain collocation points

    for i=1:size(omega_collocationPoints,2)

        # Evaluates the residue in each one of the collocation points

        phi_domain += domain_residueMetric(residue_domain(
         omega_collocationPoints[1:input_dimensionality,i], driver_input))

    end

    # Iterates through the sets of boundary points

    for i=1:length(dOmega_collocationPoints)

        # Initializes the error in this set of boundary points

        error_boundarySet = zero(T)

        # Evaluates the error in each one of the points of this set

        for j=1:size(dOmega_collocationPoints[i],2)

            # Evaluates the error

            error_boundarySet += boundary_residueMetric(dirichlet_error(
             dOmega_collocationPoints[i][1:input_dimensionality,j], 
             driver_input, dOmega_collocationPoints[i][(
             input_dimensionality+1):(input_dimensionality+
             output_dimensionality),j]))

        end

        # Multiplies the error of this set by the Lagrange multiplier of
        # this set and adds it to the phi_boundary

        phi_boundary[i] += error_boundarySet

    end

    # Returns the loss

    return (dot([phi_domain; phi_boundary], lagrange_multipliers),
     phi_boundary)

end