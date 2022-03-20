abstract type Guard{T} end


# ******************************************************************************
# T
# ******************************************************************************
struct GuardT{T} <: Guard{T}
    T :: AbstractVector{T}
    W :: AbstractVector{T}
end


function Guard(grid::GridT{T}, field) where T
    @unpack arch, wu, t, w, tguard, wguard = grid

    Tguard = guard_window_both(t, tguard)
    Tguard = adapt_array(arch, Tguard)

    Wguard = @. exp(-((w * wu)^2 / wguard^2)^20)
    Wguard = adapt_array(arch, Wguard)

    return GuardT{T}(Tguard, Wguard)
end


function apply_guard_real_domain!(E::AbstractVector, guard::GuardT)
    @. E = E * guard.T
    return nothing
end


function apply_guard_spec_domain!(E::AbstractVector, guard::GuardT)
    @. E = E * guard.W
    return nothing
end


# ******************************************************************************
# RT
# ******************************************************************************
struct GuardRT{T} <: Guard{T}
    R :: AbstractVector{T}
    T :: AbstractVector{T}
    K :: AbstractMatrix{T}
    W :: AbstractVector{T}
end


function Guard(grid::GridRT{T}, field) where T
    @unpack arch, Nr, Nt, wu, ku, r, t, k, w = grid
    @unpack rguard, tguard, kguard, wguard = grid

    Rguard = guard_window_right(r, rguard)
    Rguard = adapt_array(arch, Rguard)

    Tguard = guard_window_both(t, tguard)
    Tguard = adapt_array(arch, Tguard)

    Kguard = zeros((Nr, Nt))
    for it=1:Nt
        kmax = k_func(w[it] * wu) * sind(kguard)
        if kmax != 0
            for ir=1:Nr
                Kguard[ir, it] = exp(-((k[ir] * ku)^2 / kmax^2)^20)
            end
        end
    end
    Kguard = adapt_array(arch, Kguard)

    Wguard = @. exp(-((w * wu)^2 / wguard^2)^20)
    Wguard = adapt_array(arch, Wguard)

    return GuardRT{T}(Rguard, Tguard, Kguard, Wguard)
end


function apply_guard_real_domain!(E::AbstractMatrix, guard::GuardRT)
    guard_mul!(E, guard.R; dim=1)
    guard_mul!(E, guard.T; dim=2)
    return nothing
end


function apply_guard_spec_domain!(E::AbstractMatrix, guard::GuardRT)
    @. E = E * guard.K
    guard_mul!(E, guard.W; dim=2)
    return nothing
end


# ******************************************************************************
# Tools
# ******************************************************************************
function guard_window_left(x::AbstractArray, width; p::Int=6)
    if width >= (x[end] - x[1])
        error("Guard width is larger or equal than the grid size.")
    end
    N = length(x)
    if width == 0
        guard = ones(N)
    else
        xloc1 = x[1]
        xloc2 = x[1] + width
        gauss1 = zeros(N)
        gauss2 = ones(N)
        for i=1:N
            if x[i] >= xloc1
                gauss1[i] = 1 - exp(-((x[i] - xloc1) / (width / 2))^p)
            end
            if x[i] <= xloc2
                gauss2[i] = exp(-((x[i] - xloc2) / (width / 2))^p)
            end
        end
        guard = @. (gauss1 + gauss2) / 2
    end
    return guard
end


function guard_window_right(x::AbstractArray, width; p::Int=6)
    if width >= (x[end] - x[1])
        error("Guard width is larger or equal than the grid size.")
    end
    N = length(x)
    if width == 0
        guard = ones(N)
    else
        xloc1 = x[end] - width
        xloc2 = x[end]
        gauss1 = ones(N)
        gauss2 = zeros(N)
        for i=1:N
            if x[i] >= xloc1
                gauss1[i] = exp(-((x[i] - xloc1) / (width / 2))^p)
            end
            if x[i] <= xloc2
                gauss2[i] = 1 - exp(-((x[i] - xloc2) / (width / 2))^p)
            end
        end
        guard = @. (gauss1 + gauss2) / 2
    end
    return guard
end


function guard_window_both(x::AbstractArray, width; p::Int=6)
    if width >= (x[end] - x[1]) / 2
        error("Guard width is larger or equal than the grid size.")
    end
    lguard = guard_window_left(x, width; p=p)
    rguard = guard_window_right(x, width; p=p)
    return @. lguard + rguard - 1
end


# ------------------------------------------------------------------------------
function guard_mul!(E::Matrix, A::Vector; dim::Int=1)
    ci = CartesianIndices(size(E))
    for ici=1:length(ci)
        idim = ci[ici][dim]
        E[ici] = E[ici] * A[idim]
    end
    return nothing
end


function guard_mul!(E::CuMatrix, A::CuVector; dim::Int=1)
    N = length(E)
    ckernel = @cuda launch=false guard_mul_kernel(E, A, dim)
    config = launch_configuration(ckernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    ckernel(E, A, dim; threads=threads, blocks=blocks)
    return nothing
end


function guard_mul_kernel(E, A, dim)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    ci = CartesianIndices(size(E))
    for ici=id:stride:length(ci)
        idim = ci[ici][dim]
        E[ici] = E[ici] * A[idim]
    end
    return nothing
end
