# ---------------------------------------------------------------------------
# Ts (timestamps) and Tsd (timestamped data)
# ---------------------------------------------------------------------------

abstract type AbstractTimeSeries end

"""
    Ts(t; time_support = nothing)

Event timestamps in seconds (one spike train, stimulus onsets, licks...).

Timestamps are sorted on construction. Without `time_support`, the support is
`[first(t), last(t)]`. With one, timestamps outside it are dropped.
"""
struct Ts <: AbstractTimeSeries
    t::Vector{Float64}
    time_support::IntervalSet
    Ts(t::Vector{Float64}, ep::IntervalSet, ::Val{:unchecked}) = new(t, ep)
end

function Ts(t::AbstractVector{<:Real}; time_support::Union{Nothing,IntervalSet} = nothing)
    tt = collect(Float64, t)
    issorted(tt) || sort!(tt)
    if time_support === nothing
        ep = isempty(tt) ? IntervalSet() : IntervalSet(tt[1], tt[end])
        return Ts(tt, ep, Val(:unchecked))
    end
    return Ts(tt[_index_in(tt, time_support)], time_support, Val(:unchecked))
end

"""
    Tsd(t, d; time_support = nothing, columns = nothing)

Timestamped data. `d` can be a vector, a matrix (one column per channel or unit,
pynapple's `TsdFrame`) or an N-d array (pynapple's `TsdTensor`); time is always
the first dimension, so `size(d, 1) == length(t)`.

`columns` optionally names the columns of a matrix.
"""
struct Tsd{A<:AbstractArray} <: AbstractTimeSeries
    t::Vector{Float64}
    d::A
    columns::Vector{String}
    time_support::IntervalSet
    Tsd(t::Vector{Float64}, d::A, cols::Vector{String}, ep::IntervalSet, ::Val{:unchecked}) where {A} =
        new{A}(t, d, cols, ep)
end

function Tsd(t::AbstractVector{<:Real}, d::AbstractArray;
             time_support::Union{Nothing,IntervalSet} = nothing,
             columns = nothing)
    size(d, 1) == length(t) || throw(DimensionMismatch(
        "data has $(size(d, 1)) rows but there are $(length(t)) timestamps"))
    tt = collect(Float64, t)
    dd = d
    if !issorted(tt)
        p = sortperm(tt)
        tt, dd = tt[p], _rows(d, p)
    end
    cols = _make_columns(columns, dd)
    if time_support === nothing
        ep = isempty(tt) ? IntervalSet() : IntervalSet(tt[1], tt[end])
        return Tsd(tt, dd, cols, ep, Val(:unchecked))
    end
    keep = _index_in(tt, time_support)
    return Tsd(tt[keep], _rows(dd, keep), cols, time_support, Val(:unchecked))
end

function _make_columns(columns, d)
    columns === nothing && return String[]
    ndims(d) == 2 || throw(ArgumentError("column names only apply to matrix data"))
    cols = String.(string.(collect(columns)))
    length(cols) == size(d, 2) ||
        throw(DimensionMismatch("expected $(size(d, 2)) column names, got $(length(cols))"))
    return cols
end

# select rows (time points) of any N-d array
_rows(d::AbstractArray, idx) = d[idx, ntuple(_ -> Colon(), ndims(d) - 1)...]

# indices of sorted times `t` inside the epochs of `ep` (each index at most once)
function _index_in(t::AbstractVector{Float64}, ep::IntervalSet)
    idx = Int[]
    last = 0
    for (s, e) in ep
        lo = max(searchsortedfirst(t, s), last + 1)
        hi = searchsortedlast(t, e)
        if lo <= hi
            append!(idx, lo:hi)
            last = hi
        end
    end
    return idx
end

# --- accessors -------------------------------------------------------------

"""Timestamps (seconds)."""
times(x::AbstractTimeSeries) = x.t
"""The `IntervalSet` on which the object is defined."""
time_support(x::AbstractTimeSeries) = x.time_support
"""Data array of a `Tsd` (time on the first dimension)."""
Base.values(x::Tsd) = x.d
"""Column names of a matrix `Tsd` (empty if unnamed)."""
columns(x::Tsd) = x.columns

Base.length(x::AbstractTimeSeries) = length(x.t)
Base.isempty(x::AbstractTimeSeries) = isempty(x.t)
Base.firstindex(::AbstractTimeSeries) = 1
Base.lastindex(x::AbstractTimeSeries) = length(x)
Base.size(x::Tsd) = size(x.d)
Base.size(x::Tsd, i) = size(x.d, i)
Base.ndims(x::Tsd) = ndims(x.d)

# --- indexing --------------------------------------------------------------

Base.getindex(x::Ts, i::Integer) = x.t[i]
Base.getindex(x::Ts, idx::AbstractVector) = Ts(x.t[idx], x.time_support, Val(:unchecked))

# a single row returns the data at that time point
Base.getindex(x::Tsd{<:AbstractVector}, i::Integer) = x.d[i]
Base.getindex(x::Tsd, i::Integer) = _rows(x.d, i)
Base.getindex(x::Tsd, idx::AbstractVector) =
    Tsd(x.t[idx], _rows(x.d, idx), x.columns, x.time_support, Val(:unchecked))

# column selection on matrix data, by position or by name
_colindex(x::Tsd, j::Integer) = j
function _colindex(x::Tsd, j::AbstractString)
    k = findfirst(==(j), x.columns)
    k === nothing && throw(KeyError(j))
    return k
end
_colindex(x::Tsd, js::AbstractVector) = [_colindex(x, j) for j in js]

function Base.getindex(x::Tsd{<:AbstractMatrix}, ::Colon, j::Union{Integer,AbstractString})
    Tsd(x.t, x.d[:, _colindex(x, j)], String[], x.time_support, Val(:unchecked))
end
function Base.getindex(x::Tsd{<:AbstractMatrix}, ::Colon, js::AbstractVector)
    jj = _colindex(x, js)
    cols = isempty(x.columns) ? String[] : x.columns[jj]
    Tsd(x.t, x.d[:, jj], cols, x.time_support, Val(:unchecked))
end

"""
    map(f, x::Tsd)

Apply `f` elementwise to the data, keeping timestamps and time support.
"""
Base.map(f, x::Tsd) = Tsd(x.t, map(f, x.d), x.columns, x.time_support, Val(:unchecked))

# --- display ---------------------------------------------------------------

function _span_str(x::AbstractTimeSeries)
    isempty(x.t) && return "empty"
    return "$(round(x.t[1], sigdigits = 6)) → $(round(x.t[end], sigdigits = 6)) s"
end

function Base.show(io::IO, ::MIME"text/plain", x::Ts)
    print(io, "Ts with $(length(x)) timestamps ($(_span_str(x))), ",
          "support: $(length(x.time_support)) epoch(s)")
end
Base.show(io::IO, x::Ts) = print(io, "Ts(", length(x), " timestamps)")

function Base.show(io::IO, ::MIME"text/plain", x::Tsd)
    kind = ndims(x.d) == 1 ? "vector" : ndims(x.d) == 2 ? "matrix" : "$(ndims(x.d))-d array"
    print(io, "Tsd of ", join(size(x.d), "×"), " ", eltype(x.d), " ", kind,
          " ($(_span_str(x))), support: $(length(x.time_support)) epoch(s)")
    isempty(x.columns) || print(io, "\n  columns: ", join(x.columns, ", "))
end
Base.show(io::IO, x::Tsd) = print(io, "Tsd(", join(size(x.d), "×"), ")")
