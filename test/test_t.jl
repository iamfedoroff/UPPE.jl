include("air.jl")
UPPE.permittivity(w) = permittivity(w)
UPPE.permeability(w) = permeability(w)

grid = GridT(
    CPU();
    tu = 1e-15,   # [s] unit of time
    tmin = -200,   # [tu] left boundary of t grid
    tmax = 200,   # [tu] right boundary of t grid
    Nt = 2^14,   # number of points on t grid
    tguard = 20,   # [tu] the width of the lossy slab at the ends of t grid
    wguard = 1e16,   # [1/s] the cut-off angular frequency
)


lam0 = 0.8e-6   # [m] central wavelength
tau0 = 35e-15 / (2 * sqrt(log(2)))  # [s] initial pulse duration
I0 = 1e12 * 1e4   # [W/cm^2] initial intensity

w0 = 2 * pi * C0 / lam0   # central frequency
n0 = refractive_index(w0)
Iu = I0   # [W/m**2] unit of intensity
Eu = intensity2field(Iu, n0)

E = @. sqrt(I0 / Iu) *
    exp(-0.5 * (grid.t * grid.tu)^2 / tau0^2) * cos(w0 * grid.t * grid.tu)

field = Field(grid, w0, E; Eu, Iu)


ne, Pnl, J = zeros(size(field.E)), zero(field.E), zero(field.E)
response =  Response(ne, Pnl, J, nothing, nothing)

model = Model(
    grid, field, response;
    zu = 1,   # [m] unit of space in z direction
    z = 0,   # [m] initial propagation distance
    zmax = 21,   # [zu] propagation distance
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
@unpack tu, Nt, t = grid

zmax = model.zmax * model.zu
k0 = k_func(w0)
vg = group_velocity(w0)
zdisp = dispersion_length(w0, tau0)
taud = tau0 * sqrt(1 - 1im * zmax / zdisp)

Ea = @. tau0 / taud *
     exp(-0.5 * (t * tu)^2 / taud^2) *
     exp(1im * k0 * zmax - 1im * w0 * (t * tu + zmax / vg))

@test isapprox(field.E, Ea; rtol=1e-2)
