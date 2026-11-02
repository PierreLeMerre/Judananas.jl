module JudananasHDF5Ext

using Judananas
using Judananas: NWBFile, IntervalSet, TsGroup, Tsd
using HDF5

const SKIP_TOP = ("general", "specifications")

_ntype(obj) = haskey(attrs(obj), "neurodata_type") ? string(attrs(obj)["neurodata_type"]) : ""
_readstr(f, k, default = "") = haskey(f, k) ? string(read(f[k])) : default

# ---------------------------------------------------------------- scanning

function load_nwb(path::String)
    isfile(path) || throw(ArgumentError("no such file: $path"))
    h5open(path, "r") do f
        found = Tuple{String,Symbol,String}[]          # (name, kind, h5path)
        for k in keys(f)
            k in SKIP_TOP && continue
            obj = f[k]
            obj isa HDF5.Group && _scan!(found, obj)
        end
        order = Dict(:units => 1, :intervals => 2, :timeseries => 3)
        entries = sort(_make_keys(found), by = e -> order[e[2]])   # stable: file order within a kind
        NWBFile(path, _readstr(f, "identifier", basename(path)),
                _readstr(f, "session_description"), entries, Dict{String,Any}())
    end
end

function _scan!(found, g::HDF5.Group)
    nt = _ntype(g)
    p = HDF5.name(g)
    if nt == "Units"
        push!(found, (basename(p), :units, p))
    elseif nt == "TimeIntervals"
        push!(found, (basename(p), :intervals, p))
    elseif haskey(g, "data") && (haskey(g, "timestamps") || haskey(g, "starting_time"))
        _is_numeric(g["data"]) && push!(found, (basename(p), :timeseries, p))
    else
        for k in keys(g)
            obj = g[k]
            obj isa HDF5.Group && _scan!(found, obj)
        end
    end
end

function _is_numeric(d)
    d isa HDF5.Dataset || return false
    try
        return eltype(d) <: Number && ndims(d) >= 1 && length(d) > 0
    catch
        return false
    end
end

# Key = object name, or parent/name when the name is ambiguous (e.g. two
# "ElectricalSeries" under LFP and EMG) — the same keys as pynapple.
function _make_keys(found)
    names = [x[1] for x in found]
    return [(count(==(n), names) > 1 ? join(splitpath(p)[end-1:end], "/") : n, kind, p)
            for (n, kind, p) in found]
end

# ---------------------------------------------------------------- loading

function load_entry(path, kind::Symbol, h5path::String)
    h5open(path, "r") do f
        g = f[h5path]
        kind === :units      && return _load_units(f, g)
        kind === :intervals  && return _load_intervals(g)
        kind === :timeseries && return _load_timeseries(g)
        error("unknown kind $kind")
    end
end

# Read the columns of a DynamicTable with `n` rows (HDF5.jl reverses dims, so
# the row dimension is the last one). Ragged columns (`x` + `x_index`) become
# vectors of vectors, matrix columns become one vector per row. Columns that
# cannot be read (object references, compound types) are skipped.
function _table_columns(g, n; skip = ())
    md = Dict{Symbol,Vector}()
    for k in keys(g)
        (k in skip || endswith(k, "_index")) && continue
        ds = g[k]
        ds isa HDF5.Dataset || continue
        try
            eltype(ds) <: HDF5.Reference && continue
            x = read(ds)
            if haskey(g, k * "_index")
                ends = read(g[k*"_index"])
                length(ends) == n || continue
                x isa AbstractVector || continue
                st = [1; ends[1:end-1] .+ 1]
                md[Symbol(k)] = [x[a:b] for (a, b) in zip(st, ends)]
            elseif x isa AbstractVector
                length(x) == n && (md[Symbol(k)] = x)
            elseif x isa AbstractArray && size(x)[end] == n
                md[Symbol(k)] = [copy(selectdim(x, ndims(x), i)) for i in 1:n]
            end
        catch
            continue
        end
    end
    return md
end

function _electrode_table(f, eldset)
    if haskey(attrs(eldset), "table")
        try
            return f[attrs(eldset)["table"]]
        catch
        end
    end
    p = "general/extracellular_ephys/electrodes"
    return haskey(f, p) ? f[p] : nothing
end

const UNIT_SKIP = ("spike_times", "id", "electrodes", "waveform_mean", "waveform_sd",
                   "waveforms", "obs_intervals", "electrode_group")

function _load_units(f, u)
    st  = read(u["spike_times"])
    ends = read(u["spike_times_index"])
    n = length(ends)
    ids = haskey(u, "id") ? Int.(read(u["id"])) : collect(0:n-1)
    starts = [1; ends[1:end-1] .+ 1]
    spikes = [st[a:b] for (a, b) in zip(starts, ends)]

    md = _table_columns(u, n; skip = UNIT_SKIP)

    # electrode of each unit (first one if several) and its electrode-table columns
    if haskey(u, "electrodes")
        el = read(u["electrodes"])
        if haskey(u, "electrodes_index")
            e_ends = read(u["electrodes_index"])
            el = el[[1; e_ends[1:end-1] .+ 1]]
        end
        if length(el) == n
            md[:electrode] = el
            tab = _electrode_table(f, u["electrodes"])
            if tab !== nothing && haskey(tab, "id")
                nrow = length(tab["id"])
                ecols = _table_columns(tab, nrow; skip = ("id", "group"))
                for (k, v) in ecols
                    key = haskey(md, k) ? Symbol("electrode_", k) : k
                    md[key] = v[el .+ 1]                 # 0-based rows in NWB
                end
            end
        end
    end
    haskey(md, :rate) && (md[:rate_nwb] = pop!(md, :rate))   # :rate is computed by TsGroup

    return TsGroup(spikes; ids = ids, pairs(md)...)
end

function _load_intervals(g)
    s = read(g["start_time"])
    e = read(g["stop_time"])
    md = _table_columns(g, length(s); skip = ("start_time", "stop_time", "id", "timeseries"))
    try
        return IntervalSet(s, e; pairs(md)...)
    catch err
        err isa ArgumentError || rethrow()
        @warn "$(HDF5.name(g)): intervals overlap, so they were merged and their metadata dropped"
        return IntervalSet(s, e)
    end
end

function _load_timeseries(g)
    d = g["data"]
    data = HDF5.ismmappable(d) ? HDF5.readmmap(d) : read(d)
    # HDF5.jl reverses dims: NWB (time, channels) reads as (channels, time)
    data = ndims(data) == 1 ? data : PermutedDimsArray(data, ntuple(i -> ndims(data) + 1 - i, ndims(data)))
    nt = size(data, 1)
    t = if haskey(g, "timestamps")
        read(g["timestamps"])
    else
        t0 = Float64(read(g["starting_time"]))
        rate = Float64(attrs(g["starting_time"])["rate"])
        t0 .+ (0:nt-1) ./ rate
    end
    return Tsd(t, data)
end

end # module
