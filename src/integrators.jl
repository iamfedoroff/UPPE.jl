struct Integrator{F, U, P}
    func ::F
    u0 :: U
    p :: P
    k1 :: U
    k2 :: U
    k3 :: U
    k4 :: U
    utmp :: U
end


function Integrator(func, u0, p)
    k1, k2, k3, k4 = zero(u0), zero(u0), zero(u0), zero(u0)
    utmp = zero(u0)
    return Integrator(func, u0, p, k1, k2, k3, k4, utmp)
end


@adapt_structure Integrator


function rk4step!(integ, u, t, dt)
    @unpack func, p, utmp, k1, k2, k3, k4 = integ

    func(k1, u, p, t)

    @. utmp = u + dt * k1 / 2
    ttmp = t + dt / 2
    func(k2, utmp, p, ttmp)

    @. utmp = u + dt * k2 / 2
    ttmp = t + dt / 2
    func(k3, utmp, p, ttmp)

    @. utmp = u + dt * k3
    ttmp = t + dt
    func(k4, utmp, p, ttmp)

    @. u = u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return nothing
end


function sbe_rk4step!(integ, u, t, dt, E)
    @unpack func, p, utmp, k1, k2, k3, k4 = integ

    func(k1, u, p, t, E)

    @. utmp = u + dt * k1 / 2
    ttmp = t + dt / 2
    func(k2, utmp, p, ttmp, E)

    @. utmp = u + dt * k2 / 2
    ttmp = t + dt / 2
    func(k3, utmp, p, ttmp, E)

    @. utmp = u + dt * k3
    ttmp = t + dt
    func(k4, utmp, p, ttmp, E)

    @. u = u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return nothing
end
