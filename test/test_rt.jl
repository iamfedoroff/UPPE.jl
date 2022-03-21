include("air.jl")
UPPE.permittivity(w) = permittivity(w)
UPPE.permeability(w) = permeability(w)

grid = GridRT(
    GPU();
    ru = 1e-3,   # [m] unit of space in r direction
    rmax = 10,   # [ru] area in spatial domain
    Nr = 500,   # number of points in spatial domain
    rguard = 0,   # [ru] the width of the lossy slab at the end of r grid
    kguard = 90,   # [degrees] the cut-off angle for wave vectors
    tu = 1e-15,   # [s] unit of time
    tmin = -200,   # [tu] left boundary of t grid
    tmax = 200,   # [tu] right boundary of t grid
    Nt = 2^14,   # number of points on t grid
    tguard = 20,   # [tu] the width of the lossy slab at the ends of t grid
    wguard = 1e16,   # [1/s] the cut-off angular frequency
)


lam0 = 0.8e-6   # [m] central wavelength
a0 = 1e-3   # [m] initial beam radius
tau0 = 35e-15 / (2 * sqrt(log(2)))  # [s] initial pulse duration
I0 = 2e12 * 1e4   # [W/cm^2] initial intensity

w0 = 2 * pi * C0 / lam0   # central frequency
n0 = refractive_index(w0)
Iu = I0   # [W/m**2] unit of intensity
Eu = intensity2field(Iu, n0)

E = zeros((grid.Nr, grid.Nt))
for it=1:grid.Nt
for ir=1:grid.Nr
    E[ir,it] = sqrt(I0 / Iu) *
               exp(-0.5 * (grid.r[ir] * grid.ru)^2 / a0^2) *
               exp(-0.5 * (grid.t[it] * grid.tu)^2 / tau0^2) *
               cos(w0 * grid.t[it] * grid.tu)
end
end

field = Field(grid, w0, E; Eu, Iu)


ne = adapt_array(grid.arch, zeros(Float32, size(field.E)))
Pnl = zero(field.E)
J = zero(field.E)
response =  Response(ne, Pnl, J, nothing, nothing)

model = Model(
    grid, field, response;
    zu = 1,   # [m] unit of space in z direction
    z = 0,   # [m] initial propagation distance
    zmax = 7,   # [zu] propagation distance
    kparaxial = true,
    nonlinearity = false,
)

simulation = Simulation(
    model, n2, N0;
    prefix = "results/",
    dz0 = 1,   # [zu] initial z step
    dzout = model.zmax,   # [zu] z step for writing field into output file
)

run!(simulation)


# Compare to analytic solution:
E = collect(field.E)
@unpack ru, tu, Nr, Nt, r, t = grid

zmax = model.zmax * model.zu
k0 = k_func(w0)
vg = group_velocity(w0)
zdiff = diffraction_length(w0, a0)
zdisp = dispersion_length(w0, tau0)
taud = tau0 * sqrt(1 - 1im * zmax / zdisp)
D = 1 + 1im * zmax / zdiff

Ea = zeros(ComplexF64, (Nr, Nt))
for it=1:Nt
for ir=1:Nr
    Ea[ir,it] = tau0 / taud / D *
                exp(-0.5 * (r[ir] * ru)^2 / a0^2 / D) *
                exp(-0.5 * (t[it] * tu)^2 / taud^2) *
                exp(1im * k0 * zmax - 1im * w0 * (t[it] * tu + zmax / vg))
end
end

@test isapprox(E, Ea; rtol=1e-2)
