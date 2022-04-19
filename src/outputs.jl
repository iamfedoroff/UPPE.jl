# ******************************************************************************
# TXT
# ******************************************************************************
struct PlotVar{S, T}
    name :: S
    siunit :: S
    unit :: T
end


struct OutputTXT{S}
    fname :: S
end


# ------------------------------------------------------------------------------
# TXT T
# ------------------------------------------------------------------------------
function OutputTXT(fname::String, grid::GridT, field::FieldT, neu, zu)
    (; tu) = grid
    (; Iu) = field
    plotvars = (
        PlotVar("z", "m", zu),
        PlotVar("Imax", "W/m^2", Iu),
        PlotVar("nemax", "1/m^3", neu),
        PlotVar("tau", "s", tu),
        PlotVar("F", "J/m^2", tu * Iu),
    )
    write_txt_header(fname, plotvars)
    return OutputTXT(fname)
end


function write_txt(out::OutputTXT, analyzer::FieldAnalyzerT)
    (; z, Imax, nemax, tau, F) = analyzer
    fp = open(out.fname, "a")
    write(fp, "  ")
    @printf(fp, "%18.12e ", z)
    @printf(fp, "%18.12e ", Imax)
    @printf(fp, "%18.12e ", nemax)
    @printf(fp, "%18.12e ", tau)
    @printf(fp, "%18.12e ", F)
    write(fp, "\n")
    close(fp)
    return nothing
end


# ------------------------------------------------------------------------------
# TXT RT
# ------------------------------------------------------------------------------
function OutputTXT(fname::String, grid::GridRT, field::FieldRT, neu, zu)
    (; ru, tu) = grid
    (; Iu) = field
    plotvars = (
        PlotVar("z", "m", zu),
        PlotVar("Imax", "W/m^2", Iu),
        PlotVar("Fmax", "J/m^2", tu * Iu),
        PlotVar("nemax", "1/m^3", neu),
        PlotVar("rad", "m", ru),
        PlotVar("tau", "s", tu),
        PlotVar("W", "J", tu * ru^2 * Iu),
    )
    write_txt_header(fname, plotvars)
    return OutputTXT(fname)
end


function write_txt(out::OutputTXT, analyzer::FieldAnalyzerRT)
    (; z, Imax, Fmax, nemax, rad, tau, W) = analyzer
    fp = open(out.fname, "a")
    write(fp, "  ")
    @printf(fp, "%18.12e ", z)
    @printf(fp, "%18.12e ", Imax)
    @printf(fp, "%18.12e ", Fmax)
    @printf(fp, "%18.12e ", nemax)
    @printf(fp, "%18.12e ", rad)
    @printf(fp, "%18.12e ", tau)
    @printf(fp, "%18.12e ", W)
    write(fp, "\n")
    close(fp)
    return nothing
end


# ------------------------------------------------------------------------------
# TXT tools
# ------------------------------------------------------------------------------
function write_txt_header(fname, plotvars)
    fp = open(fname, "w")

    # write names:
    write(fp, "#")
    for pvar in plotvars
        @printf(fp, " %-18s", pvar.name)
    end
    write(fp, "\n")

    # write SI units:
    write(fp, "#")
    for pvar in plotvars
        @printf(fp, " %-18s", pvar.siunit)
    end
    write(fp, "\n")

    # write dimensionless units:
    write(fp, "#")
    for pvar in plotvars
        @printf(fp, " %-18s", pvar.unit)
    end
    write(fp, "\n")

    close(fp)
    return nothing
end


# ******************************************************************************
# BIN
# ******************************************************************************
mutable struct OutputBIN{T}
    fname :: String
    iout :: Int
    zout :: T
    dzout :: T
    zdata :: Bool
    izdata :: Int
end


# ------------------------------------------------------------------------------
# BIN T
# ------------------------------------------------------------------------------
function OutputBIN(fname::String, grid::GridT{T}, zu, z, dzout) where T
    (; tu, t) = grid

    fp = HDF5.h5open(fname, "w")
    group = HDF5.create_group(fp, "units")
    group["t"] = tu
    group["z"] = zu
    group = HDF5.create_group(fp, "grid")
    group["t"] = collect(t)
    group = HDF5.create_group(fp, "field")
    HDF5.close(fp)

    iout = 1
    zout = z

    zdata = false
    izdata = 1
    return OutputBIN{T}(fname, iout, zout, dzout, zdata, izdata)
end


function write_field(field::FieldT, group, dset)
    group[dset] = collect(field.E)
    return nothing
end


# ------------------------------------------------------------------------------
# BIN RT
# ------------------------------------------------------------------------------
function OutputBIN(fname::String, grid::GridRT{T}, zu, z, dzout) where T
    (; Nr, Nt, ru, tu, r, t) = grid
    Nwr = iseven(Nt) ? div(Nt, 2) : div(Nt+1, 2)

    fp = HDF5.h5open(fname, "w")
    group = HDF5.create_group(fp, "units")
    group["r"] = ru
    group["t"] = tu
    group["z"] = zu
    group = HDF5.create_group(fp, "grid")
    group["r"] = r
    group["t"] = collect(t)
    group = HDF5.create_group(fp, "field")
    group = HDF5.create_group(fp, "zdata")
    create_dataset(group, "z", T, ((1,), (-1,)), chunk=(100,))
    create_dataset(group, "Fr", T, ((1,Nr), (-1,Nr)), chunk=(100,Nr))
    create_dataset(group, "Si", T, ((1,Nwr+1), (-1,Nwr+1)), chunk=(100,Nwr+1))
    create_dataset(group, "ne", T, ((1,Nr), (-1,Nr)), chunk=(100,Nr))
    HDF5.close(fp)

    iout = 1
    zout = z

    zdata = true
    izdata = 1
    return OutputBIN{T}(fname, iout, zout, dzout, zdata, izdata)
end


function write_field(field::FieldRT, group, dset)
    group[dset, shuffle=true, deflate=9] = real.(collect(field.E))
    return nothing
end


function write_zdata(analyzer::FieldAnalyzerRT, group, iz)
    data = group["z"]
    HDF5.set_extent_dims(data, (iz,))
    data[iz] = analyzer.z

    data = group["Fr"]
    HDF5.set_extent_dims(data, (iz, length(analyzer.Fr)))
    data[iz, :] = collect(analyzer.Fr)

    data = group["Si"]
    Nt = length(analyzer.Si)
    Nwr = iseven(Nt) ? div(Nt, 2) : div(Nt+1, 2)
    HDF5.set_extent_dims(data, (iz, Nwr+1))
    data[iz, :] = collect(analyzer.Si)[Nwr:end]

    data = group["ne"]
    HDF5.set_extent_dims(data, (iz, length(analyzer.ne)))
    data[iz, :] = collect(analyzer.ne)
    return nothing
end



# ------------------------------------------------------------------------------
# BIN tools
# ------------------------------------------------------------------------------
function write_bin(out::OutputBIN, field::Field, analyzer::FieldAnalyzer, z)
    if z >= out.zout
        dset = @sprintf("%03d", out.iout)
        @printf("Writing dataset %s...\n", dset)

        fp = HDF5.h5open(out.fname, "r+")
        group = fp["field"]
        write_field(field, group, dset)
        HDF5.attributes(group[dset])["z"] = z
        HDF5.close(fp)

        out.iout = out.iout + 1
        out.zout = out.zout + out.dzout
    end

    if out.zdata
        fp = HDF5.h5open(out.fname, "r+")
        group = fp["zdata"]
        write_zdata(analyzer, group, out.izdata)
        HDF5.close(fp)

        out.izdata = out.izdata + 1
    end
    return nothing
end



# ******************************************************************************
# TXT + BIN
# ******************************************************************************
struct Output{TT<:OutputTXT, TB<:OutputBIN}
    txt :: TT
    bin :: TB
end


function Output(
    grid::Grid, field::Field, neu, zu, z, dzout;
    prefix::String="",
)
    if dirname(prefix) != ""
        mkpath(dirname(prefix))
    end

    fname = joinpath(dirname(prefix), basename(prefix) * "out.dat")
    txt = OutputTXT(fname, grid, field, neu, zu)

    fname = joinpath(dirname(prefix), basename(prefix) * "out.h5")
    bin = OutputBIN(fname, grid, zu, z, dzout)

    return Output(txt, bin)
end


function write_output(out::Output, field::Field, analyzer::FieldAnalyzer, z)
    write_txt(out.txt, analyzer)
    write_bin(out.bin, field, analyzer, z)
    return nothing
end
