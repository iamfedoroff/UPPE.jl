struct Simulation{T, TM, TI, TA, TO}
    model :: TM
    q_integ :: TI
    analyzer :: TA
    output :: TO
    dz0 :: T
    phik :: T
    phip :: T
    phimax :: T
    Icrit :: T
end


function Simulation(
    model, n2, N0;
    prefix="", dz0, dzout=dz0, phimax=pi/100, Icrit=Inf, alg=RK4(),
)
    (; grid, field, response, zu, z) = model

    # integrator:
    # use initial J=0 as initial condition to avoid creating dummy array
    q_prob = ODEIntegrators.Problem(q_func!, response.J, (model,))
    q_integ = ODEIntegrators.Integrator(q_prob, alg)

    analyzer = FieldAnalyzer(grid, z)

    output = Output(grid, field, N0, zu, z, dzout; prefix)

    phik = phi_kerr(field, n2, zu)
    phip = phi_plasma(field, N0, zu)

    T = float_type(grid.arch)
    TM = typeof(model)
    TI = typeof(q_integ)
    TA = typeof(analyzer)
    TO = typeof(output)
    return Simulation{T, TM, TI, TA, TO}(
        model, q_integ, analyzer, output, dz0, phik, phip, phimax, Icrit,
    )
end


function run!(simulation)
    (; model, q_integ, analyzer, output) = simulation
    (; dz0, phik, phip, phimax, Icrit) = simulation
    (; grid, field, response, z, zmax, nonlinearity) = model

    # calculate response to find the initial electron density:
    if nonlinearity
        calculate_response!(response, field.E, z)
    end

    analyze!(analyzer, grid, field, response, z)
    write_output(output, field, analyzer, z)

    dz = dz0
    zfirst = true

    while abs(z - zmax) > dz / 2
        if nonlinearity
            dzk = phimax / (phik * analyzer.Imax)
            dzp = phimax / (phip * analyzer.nemax)
            dz = min(dz0, dzk, dzp)
        end
        z = z + dz

        time2frequency!(field.E, field.FFT)
        if nonlinearity
            @timeit "Q step" ODEIntegrators.step!(q_integ, field.E, z, dz)
        end
        @timeit "K step" K_step!(field, model.KK, dz)
        frequency2time!(field.E, field.FFT)
        apply_guard_real_domain!(field.E, model.guard)

        @timeit "output" begin
            analyze!(analyzer, grid, field, response, z)
            write_output(output, field, analyzer, z)
        end

        # Exclude the first step from the timings:
        if zfirst
            reset_timer!(get_defaulttimer())
            zfirst = false
        end

        if analyzer.Imax >= Icrit
            @warn "Imax >= Icrit"
            break
        end
    end

    println(get_defaulttimer())
end


function K_step!(field::FieldT, KK, dz)
    @. field.E = field.E * exp(1im * KK * dz)
    return nothing
end


function K_step!(field::FieldRT, KK, dz)
    space2frequency!(field.E, field.DHT)
    @. field.E = field.E * exp(1im * KK * dz)
    frequency2space!(field.E, field.DHT)
    return nothing
end


function phi_kerr(field, n2, zu)
    (; w0, Eu) = field

    mu = permeability(w0)
    k0 = k_func(w0)
    chi3 = chi3_func(w0, n2)

    QQ0 = MU0 * mu * w0^2 / (2 * k0) * zu / Eu
    Rk0 = EPS0 * chi3 * 3 / 4 * Eu^3
    return QQ0 * abs(real(Rk0))
end


function phi_plasma(field, neu, zu)
    (; w0, Eu) = field

    mu = permeability(w0)
    k0 = k_func(w0)

    QQ0 = MU0 * mu * w0^2 / (2 * k0) * zu / Eu
    Rp0 = 1im / w0 * QE^2 / ME / (-1im * w0) * neu * Eu
    return QQ0 * abs(real(Rp0))
end
