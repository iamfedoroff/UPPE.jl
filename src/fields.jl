abstract type Field{T} end


# ******************************************************************************
# T
# ******************************************************************************
struct FieldT{T, TFFT} <: Field{T}
    Eu :: T
    Iu :: T
    w0 :: T
    E :: AbstractVector{Complex{T}}
    FFT :: TFFT
end


function Field(grid::GridT{T}, w0, E; Eu=1, Iu=1) where T
    (; arch) = grid

    E = Array{Complex{T}}(E)
    E = adapt_array(arch, E)

    FFT = plan_fft!(E)
    rsig2asig!(E, FFT)   # real signal -> analytic signal
    return FieldT{T, typeof(FFT)}(Eu, Iu, w0, E, FFT)
end


# ******************************************************************************
# RT
# ******************************************************************************
struct FieldRT{T, TDHT, TFFT} <: Field{T}
    Eu :: T
    Iu :: T
    w0 :: T
    E :: AbstractMatrix{Complex{T}}
    DHT :: TDHT
    FFT :: TFFT
end


function Field(grid::GridRT{T}, w0, E; Eu=1, Iu=1) where T
    (; arch, Nr, Nt, rmax) = grid

    E = Array{Complex{T}}(E)
    E = adapt_array(arch, E)

    Nwr = iseven(Nt) ? div(Nt, 2) : div(Nt+1, 2)
    region = (Nr, Nwr)
    DHT = plan_dht(rmax, E; region=region, save=false)

    FFT = plan_fft!(E, [2])
    rsig2asig!(E, FFT)   # real signal -> analytic signal

    return FieldRT{T, typeof(DHT), typeof(FFT)}(Eu, Iu, w0, E, DHT, FFT)
end



# ******************************************************************************
# Tools
# ******************************************************************************
function field2intensity(E, n0)
    return real(n0) * EPS0 * C0 * abs2(E) / 2
end


function intensity2field(I, n0)
    return sqrt(I / (real(n0) * EPS0 * C0 / 2))
end


function space2frequency!(E, DHT)
    DHT * E
    return nothing
end


function frequency2space!(E, DHT)
    DHT \ E
    return nothing
end


function time2frequency!(E, FFT)
    FFT \ E   # time -> frequency [exp(-i*w*t)]
    return nothing
end


function frequency2time!(E, FFT)
    FFT * E  # frequency -> time [exp(-i*w*t)]
    return nothing
end
