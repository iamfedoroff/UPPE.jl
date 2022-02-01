abstract type Grid{T} end


# ******************************************************************************
# T
# ******************************************************************************
struct GridT{T} <: Grid{T}
    arch :: ARCH{T}
    # number of grid points:
    Nt :: Int
    # units:
    tu :: T
    wu :: T
    # domain size:
    tmin :: T
    tmax :: T
    # grid points:
    t :: StepRangeLen{T}
    w :: AbstractVector{T}
    # grid spacing:
    dt :: T
    # Guards:
    tguard :: T
    wguard :: T
end


function GridT(
    arch::ARCH{T}=CPU(); tu=1, tmin, tmax, Nt, tguard=0, wguard=0,
) where T
    wu = 1 / tu
    t, dt, w = _grid_rectangular(tmin, tmax, Nt)
    return GridT{T}(arch, Nt, tu, wu, tmin, tmax, t, w, dt, tguard, wguard)
end


# function show(io::IO, g::GridT{T}) where T
#     str =
#     """
#     GridT{$T}:
#         Nt = $(g.Nt) - resolution in t
#         tu = $(fmt(g.tu)) [s] - unit of t
#       tmin = $(fmt(g.tmin)) [tu] - left boundary of t domain
#       tmax = $(fmt(g.tmax)) [tu] - right boundary of t domain
#         dt = $(fmt(g.dt)) [tu] - t step
#     """
#     print(io, str)
# end


# ******************************************************************************
# RT
# ******************************************************************************
struct GridRT{T} <: Grid{T}
    arch :: ARCH{T}
    # number of grid points:
    Nr :: Int
    Nt :: Int
    # units:
    ru :: T
    tu :: T
    ku :: T
    wu :: T
    # domain size:
    rmax :: T
    tmin :: T
    tmax :: T
    # Grid points:
    r :: AbstractVector{T}
    t :: StepRangeLen{T}
    k :: AbstractVector{T}
    w :: AbstractVector{T}
    # Grid spacing:
    dr :: AbstractVector{T}
    dt :: T
    # Guards:
    rguard :: T
    tguard :: T
    kguard :: T
    wguard :: T
end


function GridRT(
    arch::ARCH{T}=CPU();
    ru=1, rmax, Nr, rguard=0, kguard=90,
    tu=1, tmin, tmax, Nt, tguard=0, wguard=0,
) where T
    ku = 1 / ru
    r, dr, k = _grid_axial(rmax, Nr)

    wu = 1 / tu
    t, dt, w = _grid_rectangular(tmin, tmax, Nt)

    return GridRT{T}(
        arch, Nr, Nt, ru, tu, ku, wu, rmax, tmin, tmax, r, t, k, w, dr, dt,
        rguard, tguard, kguard, wguard,
    )
end


# ******************************************************************************
# Tools
# ******************************************************************************
function _grid_rectangular(tmin, tmax, Nt)
    # t = range(tmin, tmax, length=Nt)   # grid points

    # Exploit the fact that the Fourier transform of even real functions is
    # real. As a result, there are less noise in the corresponding spectra.
    # https://discourse.julialang.org/t/fft-function-does-not-return-the-correct-analytic-result/50644/6
    t = range(tmin, tmax, length=Nt+1)
    t = t[1:Nt]

    dt = t[2] - t[1]
    w = 2 * pi * fftfreq(Nt, 1/dt)   # angular frequency
    return t, dt, w
end


function _grid_axial(rmax, Nr)
    r = dhtcoord(rmax, Nr)
    v = dhtfreq(rmax, Nr)
    k = 2 * pi * v   # angular frequency

    # nonuniform steps:
    Nr = length(r)
    dr = zeros(Nr)
    for i=1:Nr
        dr[i] = _step(i, r)
    end
    return r, dr, k
end


"""Calculates step dx for a specific index i on a nonuniform grid x."""
function _step(i::Int, x::AbstractArray)
    Nx = length(x)
    if i == 1
        dx = x[2] - x[1]
    elseif i == Nx
        dx = x[Nx] - x[Nx - 1]
    else
        dx = (x[i+1] - x[i-1]) / 2
    end
    return dx
end
