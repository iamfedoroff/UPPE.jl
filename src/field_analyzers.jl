abstract type FieldAnalyzer end


# ******************************************************************************
# T
# ******************************************************************************
mutable struct FieldAnalyzerT{T} <: FieldAnalyzer
    # scalar variables:
    z :: T
    Imax :: T
    nemax :: T
    tau :: T
    F :: T
    # storage arrays:
    I :: AbstractVector{T}
end


function FieldAnalyzer(grid::GridT{T}, z) where T
    @unpack arch, Nt = grid
    Imax, nemax, tau, F = [0 for i=1:4]
    I = adapt_array(arch, zeros(Nt))
    return FieldAnalyzerT{T}(z, Imax, nemax, tau, F, I)
end


function analyze!(analyzer::FieldAnalyzerT, grid, field, response, z)
    @unpack t, dt = grid
    @unpack E = field
    @unpack ne = response

    @. analyzer.I = abs2(E)

    analyzer.z = z
    analyzer.Imax = maximum(analyzer.I)
    analyzer.nemax = maximum(ne)
    analyzer.tau = radius(t, collect(analyzer.I))
    analyzer.F = sum(analyzer.I) * dt

    @printf(
        "z=%18.12e[zu] Imax=%18.12e[Iu] nemax=%18.12e[neu]\n",
        analyzer.z, analyzer.Imax, analyzer.nemax,
    )
    return nothing
end


# ******************************************************************************
# RT
# ******************************************************************************
mutable struct FieldAnalyzerRT{T} <: FieldAnalyzer
    # scalar variables:
    z :: T
    Imax :: T
    nemax :: T
    rad :: T
    tau :: T
    W :: T
    # storage arrays:
    rdr :: AbstractVector{T}
    Fr :: AbstractVector{T}
    Ft :: AbstractVector{T}
end


function FieldAnalyzer(grid::GridRT{T}, z) where T
    @unpack arch, Nr, Nt, r, dr = grid
    Imax, nemax, rad, tau, W = [0 for i=1:5]
    rdr = adapt_array(arch, r .* dr)
    Fr = adapt_array(arch, zeros(Nr))
    Ft = adapt_array(arch, zeros(Nt))
    return FieldAnalyzerRT{T}(z, Imax, nemax, rad, tau, W, rdr, Fr, Ft)
end


function analyze!(analyzer::FieldAnalyzerRT, grid, field, response, z)
    @unpack r, t, dt = grid
    @unpack E = field
    @unpack ne = response

    analyzer.Fr .= vec(sum(abs2, E, dims=2)) * dt
    analyzer.Ft .= 2 * pi * vec(sum(abs2.(E) .* analyzer.rdr, dims=1))

    analyzer.z = z
    analyzer.Imax = maximum(abs2, E)
    analyzer.nemax = maximum(ne)
    analyzer.rad = radius(r, collect(analyzer.Fr))
    analyzer.tau = radius(t, collect(analyzer.Ft))
    analyzer.W = sum(analyzer.Ft) * dt

    @printf(
        "z=%18.12e[zu] Imax=%18.12e[Iu] nemax=%18.12e[neu]\n",
        analyzer.z, analyzer.Imax, analyzer.nemax,
    )
    return nothing
end


# ******************************************************************************
# Tools
# ******************************************************************************
function radius(
    x::AbstractVector{T}, y::Vector{T}; level::T=exp(-one(T)),
) where T
    Nx = length(x)
    ylevel = maximum(y) * level

    radl = zero(T)
    for i=1:Nx
        if y[i] >= ylevel
            radl = x[i]
            break
        end
    end

    radr = zero(T)
    for i=Nx:-1:1
        if y[i] >= ylevel
            radr = x[i]
            break
        end
    end

    return (abs(radl) + abs(radr)) / 2
end
