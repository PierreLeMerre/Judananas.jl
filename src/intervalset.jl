# ---------------------------------------------------------------------------
# IntervalSet: sorted, non-overlapping closed epochs [start, stop]
# ---------------------------------------------------------------------------

"""
    IntervalSet(start, stop; metadata...)
    IntervalSet(start::Real, stop::Real)
    IntervalSet(m::AbstractMatrix)      # n × 2 matrix of [start stop]
    IntervalSet()                       # empty

A set of closed epochs `[start, stop]`, kept sorted by start time.

Overlapping epochs are merged. Epochs that only touch (`stop[i] == start[i+1]`,
e.g. back-to-back trials) are kept separate. Overlapping epochs cannot carry
metadata, since merging would make the labels ambiguous.

Metadata is passed as keyword arguments, one vector per field:

```julia
ep = IntervalSet([0, 10, 20], [5, 15, 25]; condition = ["A", "B", "A"])
ep[:condition]          # ["A", "B", "A"]
for (s, e) in ep ... end
```
"""
struct IntervalSet
    start::Vector{Float64}
    stop::Vector{Float64}
    metadata::Metadata
    # unchecked constructor, for internal use on already-valid data
    IntervalSet(s::Vector{Float64}, e::Vector{Float64}, md::Metadata, ::Val{:unchecked}) =
        new(s, e, md)
end

function IntervalSet(start::AbstractVector{<:Real}, stop::AbstractVector{<:Real}; metadata...)
    length(start) == length(stop) ||
        throw(DimensionMismatch("start and stop must have the same length"))
    s = collect(Float64, start)
    e = collect(Float64, stop)
    all(e .>= s) || throw(ArgumentError("every interval needs stop >= start"))
    md = _make_metadata(metadata, length(s))

    p = sortperm(s)
    s, e = s[p], e[p]
    md = _permute_metadata(md, p)

    if _has_overlap(s, e)
        isempty(md) || throw(ArgumentError(
            "overlapping intervals cannot carry metadata; merge them first"))
        s, e = _merge_sorted(s, e; touching = false)
    end
    return IntervalSet(s, e, md, Val(:unchecked))
end

IntervalSet(start::Real, stop::Real; metadata...) = IntervalSet([start], [stop]; metadata...)
IntervalSet() = IntervalSet(Float64[], Float64[], Metadata(), Val(:unchecked))

function IntervalSet(m::AbstractMatrix{<:Real}; metadata...)
    size(m, 2) == 2 || throw(DimensionMismatch("expected an n × 2 matrix of [start stop]"))
    return IntervalSet(m[:, 1], m[:, 2]; metadata...)
end

# --- helpers ---------------------------------------------------------------

function _make_metadata(kw, n::Int)
    md = Metadata()
    for (k, v) in pairs(kw)
        vv = collect(v)
        length(vv) == n || throw(DimensionMismatch(
            "metadata field :$k has length $(length(vv)), expected $n"))
        md[Symbol(k)] = vv
    end
    return md
end

_permute_metadata(md::Metadata, p) = Metadata(k => v[p] for (k, v) in md)

_has_overlap(s, e) = any(i -> s[i+1] < e[i], 1:length(s)-1)

# Merge sorted intervals. With `touching = true`, [0,1] and [1,2] become [0,2].
function _merge_sorted(s::Vector{Float64}, e::Vector{Float64}; touching::Bool)
    isempty(s) && return s, e
    ns, ne = [s[1]], [e[1]]
    for i in 2:length(s)
        joins = touching ? s[i] <= ne[end] : s[i] < ne[end]
        if joins
            ne[end] = max(ne[end], e[i])
        else
            push!(ns, s[i]); push!(ne, e[i])
        end
    end
    return ns, ne
end

# --- basic interface -------------------------------------------------------

"""Start times of each epoch."""
starts(ep::IntervalSet) = ep.start
"""Stop times of each epoch."""
stops(ep::IntervalSet) = ep.stop
"""Metadata dictionary (field => vector)."""
metadata(ep::IntervalSet) = ep.metadata

Base.length(ep::IntervalSet) = length(ep.start)
Base.isempty(ep::IntervalSet) = isempty(ep.start)
Base.firstindex(::IntervalSet) = 1
Base.lastindex(ep::IntervalSet) = length(ep)

Base.iterate(ep::IntervalSet, i::Int = 1) =
    i > length(ep) ? nothing : ((ep.start[i], ep.stop[i]), i + 1)
Base.eltype(::Type{IntervalSet}) = Tuple{Float64,Float64}

Base.getindex(ep::IntervalSet, i::Integer) = ep[[i]]
function Base.getindex(ep::IntervalSet, idx::AbstractVector)
    s, e = ep.start[idx], ep.stop[idx]
    p = sortperm(s)     # indexing with an unsorted vector must keep the set sorted
    return IntervalSet(s[p], e[p], _permute_metadata(_permute_metadata(ep.metadata, idx), p),
                       Val(:unchecked))
end
Base.getindex(ep::IntervalSet, k::Symbol) = ep.metadata[k]
function Base.setindex!(ep::IntervalSet, v::AbstractVector, k::Symbol)
    length(v) == length(ep) || throw(DimensionMismatch("metadata length must match"))
    ep.metadata[k] = collect(v)
    return ep
end

Base.:(==)(a::IntervalSet, b::IntervalSet) =
    a.start == b.start && a.stop == b.stop && a.metadata == b.metadata

"""
    tot_length(ep)

Total duration covered by the epochs, in seconds.
"""
tot_length(ep::IntervalSet) = sum(ep.stop .- ep.start; init = 0.0)

"""`t in ep` is true when `t` falls inside any epoch (bounds included)."""
function Base.in(t::Real, ep::IntervalSet)
    i = searchsortedlast(ep.start, t)
    return i > 0 && t <= ep.stop[i]
end

# --- interval algebra (metadata is dropped by these operations) -----------

"""
    intersect(a::IntervalSet, b::IntervalSet)

Epochs covered by both `a` and `b`. Zero-length overlaps (touching edges) are dropped.
"""
function Base.intersect(a::IntervalSet, b::IntervalSet)
    s, e = Float64[], Float64[]
    i = j = 1
    while i <= length(a) && j <= length(b)
        lo = max(a.start[i], b.start[j])
        hi = min(a.stop[i], b.stop[j])
        if lo < hi
            push!(s, lo); push!(e, hi)
        end
        a.stop[i] < b.stop[j] ? (i += 1) : (j += 1)
    end
    return IntervalSet(s, e, Metadata(), Val(:unchecked))
end

"""
    union(a::IntervalSet, b::IntervalSet)

Epochs covered by `a` or `b`. Overlapping and touching epochs are merged.
"""
function Base.union(a::IntervalSet, b::IntervalSet)
    s = vcat(a.start, b.start)
    e = vcat(a.stop, b.stop)
    p = sortperm(s)
    ns, ne = _merge_sorted(s[p], e[p]; touching = true)
    return IntervalSet(ns, ne, Metadata(), Val(:unchecked))
end

"""
    setdiff(a::IntervalSet, b::IntervalSet)

Parts of `a` not covered by `b`.
"""
function Base.setdiff(a::IntervalSet, b::IntervalSet)
    s, e = Float64[], Float64[]
    j = 1
    for (as, ae) in a
        cur = as
        # skip b-epochs that end before this a-epoch starts
        while j <= length(b) && b.stop[j] <= as
            j += 1
        end
        k = j
        while k <= length(b) && b.start[k] < ae
            if b.start[k] > cur
                push!(s, cur); push!(e, b.start[k])
            end
            cur = max(cur, b.stop[k])
            k += 1
        end
        if cur < ae
            push!(s, cur); push!(e, ae)
        end
    end
    return IntervalSet(s, e, Metadata(), Val(:unchecked))
end

"""
    merge_close_intervals(ep, threshold)

Merge epochs separated by a gap of at most `threshold` seconds. Metadata is dropped.
"""
function merge_close_intervals(ep::IntervalSet, threshold::Real)
    isempty(ep) && return IntervalSet()
    ns, ne = [ep.start[1]], [ep.stop[1]]
    for i in 2:length(ep)
        if ep.start[i] - ne[end] <= threshold
            ne[end] = max(ne[end], ep.stop[i])
        else
            push!(ns, ep.start[i]); push!(ne, ep.stop[i])
        end
    end
    return IntervalSet(ns, ne, Metadata(), Val(:unchecked))
end

"""Keep only epochs longer than `threshold` seconds (metadata kept)."""
drop_short_intervals(ep::IntervalSet, threshold::Real) =
    ep[findall(ep.stop .- ep.start .> threshold)]

"""Keep only epochs shorter than `threshold` seconds (metadata kept)."""
drop_long_intervals(ep::IntervalSet, threshold::Real) =
    ep[findall(ep.stop .- ep.start .< threshold)]

# --- display ---------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", ep::IntervalSet)
    n = length(ep)
    print(io, "IntervalSet with $n epoch", n == 1 ? "" : "s",
          ", total $(round(tot_length(ep), sigdigits = 6)) s")
    allfields = sort!(collect(keys(ep.metadata)))
    fields = first(allfields, 4)          # keep rows readable
    if length(allfields) > 4
        print(io, "\n  metadata: ", join(allfields, ", "))
    end
    shown = n <= 12 ? (1:n) : vcat(1:5, n-4:n)
    for (c, i) in enumerate(shown)
        if n > 12 && c == 6
            print(io, "\n  ⋮")
        end
        print(io, "\n  [", i, "] ", ep.start[i], " → ", ep.stop[i])
        for f in fields
            print(io, "  ", f, "=", ep.metadata[f][i])
        end
    end
end
Base.show(io::IO, ep::IntervalSet) = print(io, "IntervalSet(", length(ep), " epochs)")
