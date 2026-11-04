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
    options::NamedTuple                            # load_file keywords
end

const _KIND_LABEL = Dict(:units => "TsGroup", :intervals => "IntervalSet", :timeseries => "Tsd")

Base.keys(nwb::NWBFile) = [e[1] for e in nwb.entries]
Base.haskey(nwb::NWBFile, k::AbstractString) = any(e -> e[1] == k, nwb.entries)
Base.length(nwb::NWBFile) = length(nwb.entries)

function Base.getindex(nwb::NWBFile, key::AbstractString)
    i = findfirst(e -> e[1] == key, nwb.entries)
    i === nothing && throw(KeyError(key))
    return get!(nwb.cache, key) do
        _hdf5_ext().load_entry(nwb.path, nwb.entries[i][2], nwb.entries[i][3], nwb.options)
    end
end

"""
    load_file(path; skip_columns = (), max_column_bytes = 50_000_000) -> NWBFile
    load_file(url; ...)                       # stream a remote file (experimental, see below)

Open an NWB file and list the objects Judananas can read:

- the `Units` table becomes a `TsGroup`. Every column becomes metadata, including the
  ones missing from `colnames`, and so do the columns of the electrode each unit sits on
  (`location`, coordinates...). Per-unit waveforms (`waveform_mean(s)`, `waveform_sd(s)`,
  `waveforms`) are not read.
- `TimeIntervals` tables (trials, epochs...) become `IntervalSet`s with their columns as metadata.
- every `TimeSeries` becomes a `Tsd` (time on the first dimension). Large uncompressed
  data is memory-mapped, so a 10 GB LFP is only read where you index or `restrict` it.

Data is returned as stored, with the NWB `conversion` factor not applied.

Table columns (units, electrodes, intervals) listed in `skip_columns` are not read,
nor are the ones larger than `max_column_bytes` (50 MB by default; a message names
each skipped column). Pass `max_column_bytes = Inf` to read everything.

`path` can also be an `https://` or `s3://` URL of a file on S3, such as
`dandi_url(dandiset, asset_path)`. The file is then streamed with HDF5's ROS3 driver:
only the objects you access are downloaded (keyword `region`, default `"us-east-2"`,
DANDI's region). This needs an HDF5 library built with ROS3 (`HDF5.has_ros3()`):
HDF5_jll 2.2 has it on macOS, earlier versions may not. Remote time series are read
whole when accessed. **Experimental**: opening a DANDI file with ROS3 hung in the
only test so far (macOS, HDF5_jll 2.2.3).

Needs HDF5.jl: `using Judananas, HDF5`.
"""
load_file(path::AbstractString; kwargs...) = _hdf5_ext().load_nwb(String(path); kwargs...)

const _DANDI_API = "https://api.dandiarchive.org/api"

"""
    dandi_url(dandiset, asset_path; version = "draft") -> String

The S3 URL of a file in a DANDI dataset, for `load_file` (streaming) or for downloading:

```julia
url = dandi_url("000473", "sub-216300/sub-216300_ses-216300-20200518-probe0_ecephys+ogen.nwb")
Downloads.download(url, "session.nwb")    # or load_file(url) to stream
```
"""
function dandi_url(dandiset, asset_path::AbstractString; version::AbstractString = "draft")
    ds = lpad(string(dandiset), 6, '0')
    q = "$_DANDI_API/dandisets/$ds/versions/$version/assets/?path=" * _escape_query(asset_path)
    body = String(take!(Downloads.download(q, IOBuffer())))
    m = match(r"\"asset_id\"\s*:\s*\"([0-9a-f-]+)\"", body)
    m === nothing && throw(ArgumentError("no asset $asset_path in dandiset $ds ($version)"))
    # the download endpoint redirects to a signed S3 URL; the blob itself is public
    r = Downloads.request("$_DANDI_API/assets/$(m[1])/download/"; method = "HEAD")
    return String(first(split(r.url, '?')))
end

# percent-encode everything but unreserved characters and "/" ("+" must become %2B)
_escape_query(s) = join(isascii(c) && (isletter(c) || isdigit(c) || c in "-_.~/") ? string(c) :
                        join("%" * uppercase(string(b, base = 16, pad = 2)) for b in codeunits(string(c)))
                        for c in s)

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
