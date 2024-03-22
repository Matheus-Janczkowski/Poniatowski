# Routine to store plotting functions

# Defines a function to plot a scalar field u function of two variables,
# i.e., u: R^2 -> R

function plot_surface(x_limits::Vector{Float64}, y_limits::Vector{
 Float64}, model::Function; n_pointsX=100, n_pointsY=100, x_axisName="x",
 y_axisName="y", title="u(x,y)", path_plot=joinpath(pwd(), 
 "surface_u.pdf"), domain_collocationPoints=Matrix{Float64}(undef,0,0),
 domain_pointsValues=Float64[], dirichlet_collocationPoints=Matrix{
 Float64}(undef,0,0), dirichlet_pointsValues=Float64[], 
 neumann_collocationPoints=Matrix{Float64}(undef,0,0),
 neumann_pointsValues=Float64[])

    # Creates the vector of points in x

    x_points = collect(range(start=x_limits[1], stop=x_limits[2], length=
     n_pointsX))

    y_points = collect(range(start=y_limits[1], stop=y_limits[2], length=
     n_pointsY))

    # Creates a matrix of evaluations of u 

    u_evaluations = Matrix{Float64}(undef, n_pointsX, n_pointsY)

    # Iterates through the points 

    for i=1:n_pointsX

        for j=1:n_pointsY

            # Evaluates the model

            u_evaluations[i,j] = model([x_points[i]; y_points[j]])[1]

        end 

    end

    # Makes the plot

    surface_graph = plot(x_points, y_points, u_evaluations, st=:surface,
     title=latexstring(title), xaxis=latexstring(x_axisName), yaxis=
     latexstring(y_axisName), legend=:outerbottom)

    # Adds the domain collocation points

    if size(domain_collocationPoints,2)>0

        plot!(surface_graph, domain_collocationPoints[1,:],
         domain_collocationPoints[2,:], domain_pointsValues, st=:scatter,
         color=:red, label=latexstring("Domain\\;points"))

    end

    # Adds the Dirichlet boundary points

    if size(dirichlet_collocationPoints,2)>0

        plot!(surface_graph, dirichlet_collocationPoints[1,:],
         dirichlet_collocationPoints[2,:], dirichlet_pointsValues, st=
         :scatter, color=:blue, label=latexstring("Dirichlet\\;points"))

    end

    # Adds the Neumann boundary points

    if size(neumann_collocationPoints,2)>0

        plot!(surface_graph, neumann_collocationPoints[1,:],
         neumann_collocationPoints[2,:], neumann_pointsValues, st=
         :scatter, color=:green, label=latexstring("Neumann\\;points"))

    end

    # Saves it

    savefig(surface_graph, path_plot)

end