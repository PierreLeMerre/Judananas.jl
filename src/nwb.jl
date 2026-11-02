# ---------------------------------------------------------------------------
# NWB files: `load_file(path)` works like pynapple's `nap.load_file`.
# The reading itself lives in the HDF5 extension (ext/JudananasHDF5Ext.jl),
# which loads automatically once HDF5 is loaded: `using Judananas, HDF5`.
# ---------------------------------------------------------------------------

"""
    NWBFile

What `load_file` returns: an index of the objects found in an NWB file.
Each object is read the first time you access it, and then cached.

```julia
using Judananas, HDF5
data = load_file("session.nwb")
units  = data["units"]                  # TsGroup, with every units-table column as metadata
trials = data["trials"]                 # IntervalSet, with the trial columns as metadata
lfp    = data["LFP/ElectricalSeries"]   # Tsd, memory-mapped: nothing is read until you index it
keys(data)
```
"""
struct NWBFile
    path::String
    identifier::String
    description::String
    entries::Vector{Tuple{String,Symbol,String}}   # (key, kind, path inside the file)
    cache::Dict{String,Any}
end

const _KIND_LABEL = Dict(:units => "TsGroup", :intervals => "IntervalSet", :timeseries => "Tsd")

Base.keys(nwb::NWBFile) = [e[1] for e in nwb.entries]
Base.haskey(nwb::NWBFile, k::AbstractString) = any(e -> e[1] == k, nwb.entries)
Base.length(nwb::NWBFile) = length(nwb.entries)

function Base.getindex(nwb::NWBFile, key::AbstractString)
    i = findfirst(e -> e[1] == key, nwb.entries)
    i === nothing && throw(KeyError(key))
    return get!(nwb.cache, key) do
        _hdf5_ext().load_entry(nwb.path, nwb.entries[i][2], nwb.entries[i][3])
    end
end

"""
    load_file(path) -> NWBFile

Open an NWB file and list the objects Judananas can read:

- the `Units` table becomes a `TsGroup`. Every column becomes metadata, including the
  ones missing from `colnames`, and so do the columns of the electrode each unit sits on
  (`location`, coordinates...).
- `TimeIntervals` tables (trials, epochs...) become `IntervalSet`s with their columns as metadata.
- every `TimeSeries` becomes a `Tsd` (time on the first dimension). Large uncompressed
  data is memory-mapped, so a 10 GB LFP is only read where you index or `restrict` it.

Data is returned as stored, with the NWB `conversion` factor not applied.

Needs HDF5.jl: `using Judananas, HDF5`.
"""
load_file(path::AbstractString) = _hdf5_ext().load_nwb(String(path))

function _hdf5_ext()
    ext = Base.get_extension(@__MODULE__, :JudananasHDF5Ext)
    ext === nothing && error("reading NWB files needs HDF5.jl: run `using HDF5` (once: `Pkg.add(\"HDF5\")`)")
    return ext
end

function Base.show(io::IO, ::MIME"text/plain", nwb::NWBFile)
    println(io, "NWBFile ", nwb.identifier)
    isempty(nwb.description) || println(io, "  ", nwb.description)
    w = maximum(length(e[1]) for e in nwb.entries; init = 4)
    println(io, "  ", rpad("Keys", w), "  Type")
    println(io, "  ", "─"^w, "  ", "─"^11)
    for (k, kind, _) in nwb.entries
        println(io, "  ", rpad(k, w), "  ", _KIND_LABEL[kind])
    end
end
Base.show(io::IO, nwb::NWBFile) = print(io, "NWBFile(", nwb.identifier, ", ", length(nwb), " objects)")
