# Automatic differentiation using dual numbers. This code is inspired by
# the routine created by Prof. Eduardo Lenz Cardoso (https://github.com/
# CodeLenz/LDual.jl).
#
# Some considerations:
#
# A dual number is given by a tuple of a "real" value at the first entry
# and the dual value at the second entry
#
# x = Dual(1.0, 1.0)
#
# Let a real function be evaluated using a dual argument, then, the 
# first entry corresponds to the value of the function evaluated at the
# real entry of the dual number and the second entry corresponds to the
# value of the derivative at the same point multiplied by the value of
# the dual part. 

module dual_numbers

    # Exports some basic functions from Julia

    export Dual, +, *, -,/,sin,cos, exp, log, sqrt, abs, ^

    # Exports the definitions of one and zero to extend them

    export one, zero

    # Exports a method to convert numbers to dual format

    export convert

    # Exports operations over arrays

    export transpose, dot, Rand, norm

    # Defines the basic dual type

    struct  Dual

        real::Float64

        dual::Float64

    end

    ####################################################################
    #                          Some identities                         #
    ####################################################################

    # Multiplicative identity

    import Base:one

    function one(T::Type{Dual})

        return Dual(1.0, 0.0)

    end

    # Aditive identity

    import Base:zero

    function zero(T::Type{Dual})

        return Dual(0.0, 0.0)

    end

    import Base:zero

    function zero(a::Dual)

        return Dual(0.0, 0.0)

    end

    # Converts a number to dual

    import Base:convert

    function convert(::Type{Dual},x::Number)

        return Dual(x, 0.0)

    end

    ####################################################################
    #                         Basic operations                         #
    ####################################################################

    # Let z = f(x), where z and x are dual numbers. The real part of z 
    # is the value of the function f evaluated at the real part of x,
    # while the dual part of z is the derivative of f evaluated at the
    # real part of x but multiplied by the dual part of x

    # Sum of two numbers

    import Base:+

    function +(x::Dual, y::Dual)

        p = x.real + y.real

        d = x.dual + y.dual

        return Dual(p, d)

    end

    # Product of two numbers

    import Base:*

    function *(x::Dual, y::Dual)

        p = x.real*y.real

        d = x.real*y.dual + x.dual*y.real

        return Dual(p, d)

    end

    # Negation of a number

    import Base:-

    function -(x::Dual)

        p = -x.real

        d = -x.dual

        return Dual(p, d)

    end

    # Subtraction of two numbers

    import Base:-

    function -(x::Dual,y::Dual)

        y1 = -y

        return (x + y1)

    end

    # Division between two numbers

    import Base:/

    function /(x::Dual,y::Dual)

        # Catches the case when the real part of the denominator is null

        @assert y.real!=0

        # Calculates the function and its derivative

        function_value = 1.0/y.real

        derivative_value = -1*y.dual/(y.real^2)

        # Returns the dual tuple

        return x*Dual(function_value, derivative_value)

    end

    # Sine function

    import Base:sin

    function sin(x::Dual)

        p = sin(x.real)

        d = cos(x.real)*x.dual

        return Dual(p, d)

    end

    # Cosine

    import Base:cos

    function cos(x::Dual)

        p = cos(x.real)

        d = -sin(x.real)*x.dual

        return Dual(p, d)

    end

    # Exponential

    import Base:exp

    function exp(x::Dual)

        p = exp(x.real)

        d = p*x.dual

        return Dual(p, d)

    end

    # Hyperbolic tangent

    import Base:tanh

    function tanh(x::Dual)

        e2x = exp(2*x)

        return ((e2x-1.0)/(e2x+1.0))

    end

    # Hyperbolic sine

    import Base:sinh

    function sinh(x::Dual)

        return ((exp(x)-exp(-x))/2)

    end

    # Hyperbolic cosine

    import Base:cosh

    function cosh(x::Dual)

        return ((exp(x)+exp(-x))/2)

    end

    # Natural logarithm

    import Base:log

    function log(x::Dual)

        # Avoids division by zero

        @assert x.real!=0

        p = log(x.real)

        d = x.dual/x.real

        return Dual(p, d)

    end

    # Square root

    import Base:sqrt

    function sqrt(x::Dual)

        # Avoids division by zero

        @assert x.real!=0

        p = sqrt(x.real)

        d = x.dual/(2*sqrt(x.real))

        return Dual(p, d)

    end

    # Absolute value

    import Base:abs

    function abs(x::Dual)

        # Avoids division by zero

        @assert x.real!=0

        p = abs(x.real)

        d = x.dual*(x.real/abs(x.real))

        return Dual(p, d)

    end

    ####################################################################
    #       Special cases between common numbers and dual numbers      #
    ####################################################################

    # Multiplication of dual number by non-dual number

    function *(x, y::Dual)

        # Multiplies both

        return Dual(x*y.real, x*y.dual)

    end

    function *(x::Dual, y)

        # Multiplies both

        return (y*x.real, y*x.dual)

    end

    # Sum of dual number with non-dual number

    function +(x,y::Dual)

        # Sums them
        
        return Dual(x+y.real, y.dual)

    end

    function +(x::Dual,y)

        # Sums them

        return Dual(x.real+y, x.dual)

    end

    # Subtraction of dual number by non-dual number

    function -(x,y::Dual)

        # Subtracts them

        return Dual(x-y.real, -y.dual)

    end

    function -(x::Dual,y)

        # Subtracts them

        return (x.real-y, x.dual)

    end


    ####################################################################
    #                   Vector and matrix definitions                  #
    ####################################################################

    # Transpose operation for vectors

    import LinearAlgebra:transpose

    function transpose(A::Vector{Dual})

        # Gets the dimensions of A

        dims = size(A)

        return reshape(A, 1, length(A))

    end

    # Transpose operation for matrices

    import LinearAlgebra:transpose

    function transpose(A::Matrix{Dual})

        # Gets the dimensions of A

        dims = size(A)

        return  permutedims(A, (2, 1))

    end

    # Product of a real scalar by an array

    function *(x::Float64,A::Array{Dual})

        # Initializes the output 

        V = zeros(A)

        # Aplica o produto em cada uma das posições

        for i in eachindex(A)

            V[i] = x*A[i]

        end

        return V

    end

    # Product of a dual scalar by a float array

    function *(x::Dual,A::Array{Float64})

        # Initializes the output 

        V = zeros(A)

        # Aplica o produto em cada uma das posições

        for i in eachindex(A)

            V[i] = x*A[i]

        end

        return V

    end

    function *(x::Dual,A::Array{Dual})

        # Initializes the output 

        V = zeros(A)

        # Aplica o produto em cada uma das posições

        for i in eachindex(A)

            V[i] = x*A[i]

        end

        return V

    end

    # Dot product

    import LinearAlgebra:dot

    function dot(A::Vector{Dual},B::Vector{Dual})

        return transpose(A)*B

    end

    import LinearAlgebra:dot

    function dot(A::Vector{Float64},B::Vector{Dual})

        return transpose(A)*B

    end

    import LinearAlgebra:dot

    function dot(A::Vector{Dual},B::Vector{Float64})

        return transpose(A)*B

    end

    # Rand function for arrays
    #   rand(T::Type, d1::Integer, dims::Integer...) at random.jl:232
    #   rand(T::Type{FD.dual}, dims...) at /home/lenz/Dropbox/dif_automatica.jl:245
    # VOU USAR Rand
    #import Base.rand

    function Rand(T::Type{Dual}, dims...)

        # Initializes the array

        V = Array{T}(undef,dims)

        # Initializes all the dual values as null

        for i in eachindex(V)

            V[i] = Dual(rand(), 0.0)

        end

        # Returns the array

        return V

    end

    #
    # calcula a norma 2 - Só a 2 !
    #
    import LinearAlgebra:norm
    function norm(A::Vector{Dual}, p=2)

        # Verifies wheter the asked norm is two
        p==2 || throw("LDual::norm p=2 only is implemented")

        # Converts to vector

        a = vec(A)

        # Evaluates the inner product

        prod = dot(a, a)

        # Returns the square root

        return sqrt(prod)

    end

    ####################################################################
    #                 Special cases for exponentiation                 #
    ####################################################################

    import Base:^

    # x^y, where x is not dual

    function ^(x::T,y::Dual) where T<:Number

        # Dual(((x)^(y.real)), y.dual*(((pi*im)+log(abs(x)))*(x^y.real)))

        x>=zero(T) || throw(DomainError(x, "Negative values of base fo",
         "r exponentiation are not allowed in the LDual library yet."))

        # Common evaluation

        common_calc = x^(y.real)

        # Caso dz/dy, x constante
        ifelse(x>0,  Dual(common_calc, y.dual*log(x)*common_calc), Dual(
         common_calc, 0.0))

    end 

    # x^y, where x is dual and y is not dual

    function ^(x::Dual,y::T) where T<:Number

        ifelse(y!=zero(T), Dual(((x.real)^(y)), x.dual*(y*(x.real^(y-1))
         )),  Dual(((x.real)^(y)), 0.0) )
    
    end

    # x^y, where both are dual

    function ^(x::Dual,y::Dual)

        # Avoids complex results
        #
        # Dual(((x.real)^(y.real)), (((x.real)^(y.real))*(((log(
        # abs(x.real))+(pi*im))*y.dual)+((y.real/x.real)*x.dual))))

        x.real >= 0 || throw(DomainError(x, "Negative values of base f",
         "or exponentiation are not allowed in the LDual library yet."))  

        # Caso x e y variáveis, ou x^x
    
        common_calc = x.real^y.real

        ifelse(x.real>0, Dual(common_calc, common_calc*((log(x.real)*y.
         dual)+((y.real/x.real)*x.dual))), Dual(common_calc, 0.0) )
    
    end

end 
