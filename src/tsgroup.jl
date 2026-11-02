# ---------------------------------------------------------------------------
# TsGroup: a population of spike trains sharing one time support
# ---------------------------------------------------------------------------

"""
    TsGroup(spikes; ids = 1:n, time_support = nothing, metadata...)
    TsGroup(d::AbstractDict)          # id => spike times

A group of `Ts` (typically the units of one session) with one shared time
support and per-unit metadata.

- `spikes`: a vector whose elements are spike-time vectors or `Ts` objects.
- `ids`: unit identifiers (e.g. Kilosort cluster ids); default `1:n`.
- `time_support`: if omitted, the union of the units' supports when `Ts` are
  given, otherwise `[earliest spike, latest spike]` across units.
- metadata: keyword vectors, one value per unit (`location = [...]`, `depth = [...]`).

The `:rate` field (spikes / s over the time support) is computed automatically
and updated by `restrict`.

Indexing is positional: `g[3]` is the third unit, whatever its id.
Use `filter` to select units from their metadata:

```julia
pfc = filter(u -> u.location == "PFC" && u.rate > 1, g)
```
"""
struct TsGroup
    units::Vector{Ts}
    ids::Vector{Int}
    metadata::Metadata
    time_support::IntervalSet
    TsGroup(u::Vector{Ts}, ids::Vector{Int}, md::Metadata, ep::IntervalSet, ::Val{:unchecked}) =
        new(u, ids, md, ep)
end

function TsGroup(spikes::AbstractVector; ids = nothing,
                 time_support::Union{Nothing,IntervalSet} = nothing, metadata...)
    n = length(spikes)
    uid = ids === nothing ? collect(1:n) : collect(Int, ids)
    length(uid) == n || throw(DimensionMismatch("expected $n ids, got $(length(uid))"))
    allunique(uid) || throw(ArgumentError("unit ids must be unique"))
    haskey(metadata, :rate) &&
        throw(ArgumentError(":rate is computed automatically and cannot be passed"))
    md = _make_metadata(metadata, n)

    ep = time_support === nothing ? _default_support(spikes) : time_support
    units = Ts[Ts(_spike_vector(s); time_support = ep) for s in spikes]
    md[:rate] = _rates(units, ep)
    return TsGroup(units, uid, md, ep, Val(:unchecked))
end

function TsGroup(d::AbstractDict; kwargs...)
    k = sort!(collect(keys(d)))
    return TsGroup([d[i] for i in k]; ids = k, kwargs...)
end

_spike_vector(s::Ts) = s.t
_spike_vector(s::AbstractVector{<:Real}) = s

function _default_support(spikes)
    isempty(spikes) && return IntervalSet()
    if all(s -> s isa Ts, spikes)
        return reduce(union, (s.time_support for s in spikes))
    end
    nonempty = [_spike_vector(s) for s in spikes if !isempty(s)]
    isempty(nonempty) && return IntervalSet()
    return IntervalSet(minimum(minimum, nonempty), maximum(maximum, nonempty))
end

function _rates(units::Vector{Ts}, ep::IntervalSet)
    T = tot_length(ep)
    return [T > 0 ? length(u) / T : NaN for u in units]
end

# --- accessors -------------------------------------------------------------

"""Unit identifiers of a `TsGroup`."""
ids(g::TsGroup) = g.ids
metadata(g::TsGroup) = g.metadata
time_support(g::TsGroup) = g.time_support
"""Firing rate of each unit over the time support (spikes / s)."""
rates(g::TsGroup) = g.metadata[:rate]

Base.length(g::TsGroup) = length(g.units)
Base.isempty(g::TsGroup) = isempty(g.units)
Base.iterate(g::TsGroup, i::Int = 1) = i > length(g) ? nothing : (g.units[i], i + 1)
Base.eltype(::Type{TsGroup}) = Ts
Base.lastindex(g::TsGroup) = length(g)

Base.getindex(g::TsGroup, i::Integer) = g.units[i]
function Base.getindex(g::TsGroup, idx::AbstractVector)
    md = Metadata(k => v[idx] for (k, v) in g.metadata)
    return TsGroup(g.units[idx], g.ids[idx], md, g.time_support, Val(:unchecked))
end
Base.getindex(g::TsGroup, k::Symbol) = g.metadata[k]
function Base.setindex!(g::TsGroup, v::AbstractVector, k::Symbol)
    k == :rate && throw(ArgumentError(":rate is computed automatically"))
    length(v) == length(g) || throw(DimensionMismatch("metadata length must match"))
    g.metadata[k] = collect(v)
    return g
end

# one metadata row as a NamedTuple, including the id
function _row(g::TsGroup, i::Int)
    ks = sort!(collect(keys(g.metadata)))
    return NamedTuple{(:id, ks...)}((g.ids[i], (g.metadata[k][i] for k in ks)...))
end

"""
    filter(f, g::TsGroup)

Keep the units for which `f(row)` is true. `row` is a NamedTuple with the unit
`id` and every metadata field, e.g. `filter(u -> u.rate > 2, g)`.
"""
Base.filter(f, g::TsGroup) = g[[i for i in 1:length(g) if f(_row(g, i))]]

# --- display ---------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", g::TsGroup)
    n = length(g)
    print(io, "TsGroup with $n unit", n == 1 ? "" : "s",
          ", support: $(length(g.time_support)) epoch(s), ",
          "$(round(tot_length(g.time_support), sigdigits = 6)) s")
    n == 0 && return
    others = sort!([k for k in keys(g.metadata) if k != :rate])
    ks = [:rate; first(others, 5)]                 # keep the table narrow
    if length(others) > 5
        print(io, "\n  metadata: rate, ", join(others, ", "))
    end
    shown = n <= 12 ? (1:n) : vcat(1:5, n-4:n)
    fmt(v) = v isa AbstractFloat ? string(round(v, sigdigits = 4)) :
             v isa AbstractArray ? "[" * join((fmt(x) for x in first(v, 3)), ", ") * (length(v) > 3 ? ", …]" : "]") :
             string(v)
    cells = [[fmt(g.metadata[k][i]) for i in shown] for k in ks]
    idw = max(2, maximum(i -> length(string(g.ids[i])), shown)) + 2
    widths = [min(18, max(length(string(k)), maximum(length, c))) + 2 for (k, c) in zip(ks, cells)]
    print(io, "\n  ", rpad("id", idw), join(rpad(string(k), w) for (k, w) in zip(ks, widths)))
    for (c, i) in enumerate(shown)
        n > 12 && c == 6 && print(io, "\n  ⋮")
        print(io, "\n  ", rpad(string(g.ids[i]), idw))
        for (j, w) in enumerate(widths)
            s = cells[j][c]
            print(io, rpad(length(s) > w - 2 ? first(s, w - 3) * "…" : s, w))
        end
    end
end
Base.show(io::IO, g::TsGroup) = print(io, "TsGroup(", length(g), " units)")
