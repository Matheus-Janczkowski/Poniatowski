# Routine to store functions for the loss function of PINNs

# Defines a function to calculate the loss function of the PINN model 
# with the domain parcel only

function phi_lossDomain(driver_input::Function, parameters::Vector{T},
 omega_collocationPoints::Matrix{Float64}, residue_domain::Function,
 domain_residueMetric::Function) where {T<:Number}

    # Initializes the loss function for each one of the parcels
    
    phi = zero(T)

    # Iterates through the domain collocation points

    for i=1:size(omega_collocationPoints,2)

        # Evaluates the residue in each one of the collocation points

        phi += domain_residueMetric(residue_domain(
         omega_collocationPoints[:,i], driver_input))

    end

    # Returns the loss

    return phi

end

# Defines a function to calculate the loss function of the PINN model 
# with the boundary parcel only

function phi_lossBoundary(driver_input::Function, parameters::Vector{T},
 dOmega_collocationDirichlet::Vector{Matrix{Float64}}, 
 dOmega_valuesDirichlet::Vector{Matrix{Float64}},
 dOmega_outputIndexesDirichlet::Vector{Matrix{Int64}},
 dOmega_collocationNeumann::Vector{Matrix{Float64}},
 dOmega_valuesNeumann::Vector{Matrix{Float64}}, dirichlet_error::
 Function, neumann_error::Function, boundary_residueMetric::Function) where {T<:Number}

    # Initializes the loss function for each one of the parcels
    
    phi = Vector{T}(undef, length(dOmega_collocationDirichlet)+length(
    dOmega_collocationNeumann))

    # Initializes a counter of boundary points

    boundary_counter = 1

    # Iterates through the sets of boundary points with Dirichlet boun-
    # dary conditions

    for i=1:length(dOmega_collocationDirichlet)

        # Initializes the error in this set of boundary points

        error_boundarySet = zero(T)

        # Evaluates the error in each one of the points of this set

        for j=1:size(dOmega_collocationDirichlet[i],2)

            # Evaluates the error

            error_boundarySet += boundary_residueMetric(dirichlet_error(
            dOmega_collocationDirichlet[i][:,j], driver_input,
             dOmega_valuesDirichlet[i][:,j], dOmega_outputIndexesDirichlet[i][:,j]))

        end

        # Adds it to the phi

        phi[boundary_counter] += error_boundarySet

        # Updates the counter

        boundary_counter += 1

    end

    # Iterates through the sets of boundary points with Neumann boundary
    # conditions

    for i=1:length(dOmega_collocationNeumann)

        # Initializes the error in this set of boundary points

        error_boundarySet = zero(T)

        # Evaluates the error in each one of the points of this set

        for j=1:size(dOmega_collocationNeumann[i],2)

            # Evaluates the error

            error_boundarySet += boundary_residueMetric(neumann_error(
            dOmega_collocationNeumann[i][:,j], driver_input,
             dOmega_valuesNeumann[i][:,j]))

        end

        # Adds it to the phi

        phi[boundary_counter] += error_boundarySet

        # Updates the counter

        boundary_counter += 1

    end

    # Returns the loss

    return phi

end