function permittivity(w)
    # E.R. Peck and K. Reeder "Dispersion of Air" JOSA, 62, 958 (1972)
    if w == 0
        sig = 0
    else
        lam = 2 * pi * C0 / w
        sig = 1 / lam
        sig = sig * 1e-6   # 1/m -> 1/um
    end
    dum = 8060.51 + 2480990 / (132.247 - sig^2) + 17455.7 / (39.32957 - sig^2)
    n = 1 + dum * 1e-8
    eps = n^2
    return eps
end


function permeability(w)
    mu = 1
    return mu
end


n2 = 0.5e-23   # [m**2/W] nonlinear index
N0 = 2.5e25   # [1/m^3] neutrals density
