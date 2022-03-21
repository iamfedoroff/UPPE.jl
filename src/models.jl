# ******************************************************************************
# Response
# ******************************************************************************
struct Response{T, F, P}
    ne :: AbstractArray{T}
    Pnl :: AbstractArray{Complex{T}}
    J :: AbstractArray{Complex{T}}
    func! :: F
    p :: P
end


function calculate_response!(resp::Response, E, z)
    resp.func!(resp.ne, resp.Pnl, resp.J, E, resp.p, z)
    return nothing
end


# ******************************************************************************
# Model
# ******************************************************************************
struct Model{T, TG, TF, TR}
    grid :: TG
    field :: TF
    response :: TR
    guard :: Guard{T}
    KK :: AbstractArray{Complex{T}}
    QQ :: AbstractArray{Complex{T}}
    QJ :: AbstractVector{Complex{T}}
    zu :: T
    z :: T
    zmax :: T
    nonlinearity :: Bool
end


function Model(
    grid::Grid{T}, field, response;
    zu=1, z=0, zmax, kparaxial=true, qparaxial=true, nonlinearity=true,
) where T
    @unpack arch, Nt, w, wu = grid

    guard = Guard(grid, field)

    # Linear propagator:
    KK = KKpropagator(grid, field, zu; paraxial=kparaxial)
    KK = adapt_array(arch, KK)
    apply_guard_spec_domain!(KK, guard)

    # Nonlinear propagator:
    QQ = QQpropagator(grid, field, zu; paraxial=qparaxial)
    QQ = adapt_array(arch, QQ)
    apply_guard_spec_domain!(QQ, guard)

    QJ = zeros(ComplexF64, Nt)   # Float64 for higher precision
    for it=1:Nt
        QJ[it] = QJfunc(w[it] * wu)
    end
    QJ = Array{Complex{T}}(QJ)
    QJ = adapt_array(arch, QJ)
    @. QJ = QJ * guard.W

    TG = typeof(grid)
    TF = typeof(field)
    TR = typeof(response)
    return Model{T, TG, TF, TR}(
        grid, field, response, guard, KK, QQ, QJ, zu, z, zmax, nonlinearity,
    )
end


# ******************************************************************************
# K propagator
# ******************************************************************************
function KKfunc(k, w; paraxial=true)
    beta = beta_func(w)
    KK = 0
    if paraxial
        if beta != 0
            KK = beta - k^2 / (2 * beta)
        end
    else
        KK = sqrt(beta^2 - k^2 + 0im)
    end
    return KK
end


function KKpropagator(grid::GridT{T}, field, zu; paraxial=true) where T
    @unpack Nt, w, wu = grid
    @unpack w0 = field
    vf = group_velocity(w0)   # frame velocity
    KK = zeros(ComplexF64, Nt)   # Float64 for higher precision
    for it=1:Nt
        if w[it] > 0
            KK[it] = KKfunc(0, w[it] * wu; paraxial) * zu
            KK[it] = KK[it] - (w[it] * wu) / vf * zu
        end
    end
    return Array{Complex{T}}(KK)
end


function KKpropagator(grid::GridRT{T}, field, zu; paraxial=true) where T
    @unpack Nr, Nt, k, ku, w, wu = grid
    @unpack w0 = field
    vf = group_velocity(w0)   # frame velocity
    KK = zeros(ComplexF64, (Nr, Nt))   # Float64 for higher precision
    for it=1:Nt
        if w[it] > 0
            for ir=1:Nr
                KK[ir,it] = KKfunc(k[ir]*ku, w[it]*wu; paraxial)
                KK[ir,it] = (KK[ir,it] - (w[it] * wu) / vf) * zu
            end
        end
    end
    return Array{Complex{T}}(KK)
end


# ******************************************************************************
# Q propagator
# ******************************************************************************
function QQfunc(k, w; paraxial=true)
    mu = permeability(w)
    beta = beta_func(w)
    QQ = 0
    if paraxial
        if beta != 0
            QQ = MU0 * mu * w^2 / (2 * beta)
        end
    else
        KK = sqrt(beta^2 - k^2 + 0im)
        if KK != 0
            QQ = MU0 * mu * w^2 / (2 * KK)
        end
    end
    return QQ
end


function QQpropagator(grid::GridT{T}, field, zu; paraxial=true) where T
    @unpack Nt, w, wu = grid
    @unpack Eu, w0 = field
    QQ = zeros(ComplexF64, Nt)   # Float64 for higher precision
    for it=1:Nt
        if w[it] > 0
            QQ[it] = QQfunc(0, w[it] * wu; paraxial) * zu / Eu
        end
    end
    return Array{Complex{T}}(QQ)
end


function QQpropagator(grid::GridRT{T}, field, zu; paraxial=true) where T
    @unpack Nr, Nt, k, ku, w, wu = grid
    @unpack Eu, w0 = field
    QQ = zeros(ComplexF64, (Nr, Nt))   # Float64 for higher precision
    for it=1:Nt
        if w[it] > 0
            for ir=1:Nr
                QQ[ir,it] = QQfunc(k[ir]*ku, w[it]*wu; paraxial) * zu / Eu
            end
        end
    end
    return Array{Complex{T}}(QQ)
end


function QJfunc(w)
    QJ = 0
    if w != 0
        QJ = 1im / w
    end
    return QJ
end


# ******************************************************************************
# Q func
# ******************************************************************************
function q_func!(dE, E, p, z)
    model, = p
    @unpack QQ, QJ, field, response, guard = model
    @unpack FFT = field
    @unpack Pnl, J = response

    frequency2time!(E, FFT)

    calculate_response!(response, E, z)

    # nonlinear polarization:
    apply_guard_real_domain!(Pnl, guard)
    rsig2aspec!(Pnl, FFT)   # real signal -> analytic spectrum
    apply_guard_spec_domain!(Pnl, guard)

    # current:
    apply_guard_real_domain!(J, guard)
    rsig2aspec!(J, FFT)   # real signal -> analytic spectrum
    apply_guard_spec_domain!(J, guard)

    dE_update!(dE, Pnl, J, QQ, QJ)   # @. dE = 1im * QQ * (Pnl + QJ * J)

    time2frequency!(E, FFT)

    return nothing
end


function dE_update!(
    dE::TA, Pnl::TA, J::TA, QQ::TA, QJ::TA,
) where TA<:AbstractVector
    @. dE = 1im * QQ * (Pnl + QJ * J)
    return nothing
end


function dE_update!(
    dE::TA, Pnl::TA, J::TA, QQ::TA, QJ::TB,
) where {TA<:AbstractMatrix, TB<:AbstractVector}
    Nr, Nt = size(dE)
    for it=1:Nt
    for ir=1:Nr
        dE[ir,it] = 1im * QQ[ir,it] * (Pnl[ir,it] + QJ[it] * J[ir,it])
    end
    end
    return nothing
end


function dE_update!(
    dE::TA, Pnl::TA, J::TA, QQ::TA, QJ::TB,
) where {TA<:CuMatrix, TB<:CuVector}
    N = length(dE)
    ckernel = @cuda launch=false dE_update_kernel(dE, Pnl, J, QQ, QJ)
    config = launch_configuration(ckernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    ckernel(dE, Pnl, J, QQ, QJ; threads=threads, blocks=blocks)
    return nothing
end


function dE_update_kernel(dE, Pnl, J, QQ, QJ)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    ci = CartesianIndices(size(dE))
    for ici=id:stride:length(ci)
        it = ci[ici][2]
        dE[ici] = 1im * QQ[ici] * (Pnl[ici] + QJ[it] * J[ici])
    end
    return nothing
end
