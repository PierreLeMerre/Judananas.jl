# ---------------------------------------------------------------------------
# Core operations on time series
# ---------------------------------------------------------------------------

# --- restrict --------------------------------------------------------------

"""
    restrict(x, ep::IntervalSet)
    restrict(x, start, stop)

Keep only the part of `x` inside `ep`. The new time support is the
intersection of `x`'s support and `ep`. Works on `Ts`, `Tsd` and `TsGroup`
(rates are recomputed for a `TsGroup`).
"""
function restrict(x::Ts, ep::IntervalSet)
    sup = intersect(x.time_support, ep)
    return Ts(x.t[_index_in(x.t, sup)], sup, Val(:unchecked))
end

function restrict(x::Tsd, ep::IntervalSet)
    sup = intersect(x.time_support, ep)
    idx = _index_in(x.t, sup)
    return Tsd(x.t[idx], _rows(x.d, idx), x.columns, sup, Val(:unchecked))
end

function restrict(g::TsGroup, ep::IntervalSet)
    sup = intersect(g.time_support, ep)
    units = Ts[Ts(u.t[_index_in(u.t, sup)], sup, Val(:unchecked)) for u in g.units]
    md = copy(g.metadata)
    md[:rate] = _rates(units, sup)
    return TsGroup(units, g.ids, md, sup, Val(:unchecked))
end

restrict(x, start::Real, stop::Real) = restrict(x, IntervalSet(start, stop))

# --- rate and inter-event intervals ------------------------------------------

"""
    rate(x::Ts)

Event rate over the time support: `length(x) / tot_length(time_support(x))`, in events / s
(pynapple's `Ts.rate`). `NaN` for an empty support. After `restrict(x, ep)`, this is the
rate inside `ep`. For a `TsGroup`, see `rates`.
"""
function rate(x::Ts)
    T = tot_length(x.time_support)
    return T > 0 ? length(x) / T : NaN
end

"""
    time_diff(x::Ts; align = :center, epochs = time_support(x))

Intervals between consecutive events (ISIs, for a spike train), as a `Tsd` (pynapple's
`Ts.time_diff`). Each interval is timed at the middle of the two events (`align = :center`),
at the first one (`:start`) or at the second one (`:end`).

Intervals are computed inside each epoch of `epochs`, never across a gap between epochs.
"""
function time_diff(x::Ts; align::Symbol = :center, epochs::IntervalSet = x.time_support)
    align in (:center, :start, :end) ||
        throw(ArgumentError("align must be :center, :start or :end"))
    t, d = Float64[], Float64[]
    for (s, e) in epochs
        lo, hi = searchsortedfirst(x.t, s), searchsortedlast(x.t, e)
        for k in lo:hi-1
            a, b = x.t[k], x.t[k+1]
            push!(t, align === :center ? (a + b) / 2 : align === :start ? a : b)
            push!(d, b - a)
        end
    end
    return Tsd(t, d, String[], epochs, Val(:unchecked))
end

_isis(x::Ts, ep) = values(time_diff(x; epochs = ep))

function _isi_edges(bins::Integer, isis, log_scale::Bool)
    bins > 0 || throw(ArgumentError("bins must be positive"))
    all_isi = reduce(vcat, isis; init = Float64[])
    log_scale && (all_isi = filter(>(0), all_isi))
    isempty(all_isi) && return collect(range(0.0, 1.0; length = bins + 1))
    lo, hi = extrema(all_isi)
    if log_scale
        lo == hi && (lo /= 10; hi *= 10)
        return exp10.(range(log10(lo), log10(hi); length = bins + 1))
    end
    lo == hi && (lo -= 0.5; hi += 0.5)
    return collect(range(lo, hi; length = bins + 1))
end
function _isi_edges(bins::AbstractVector{<:Real}, _, _)
    length(bins) >= 2 && issorted(bins) ||
        throw(ArgumentError("bin edges must be sorted, with at least 2 edges"))
    return collect(Float64, bins)
end

# counts of `v` in bins [e[i], e[i+1]); the last bin includes its right edge (like numpy)
function _histogram(v, edges)
    c = zeros(Int, length(edges) - 1)
    for x in v
        (x < edges[1] || x > edges[end]) && continue
        i = min(searchsortedlast(edges, x), length(c))
        c[i] += 1
    end
    return c
end

"""
    compute_isi_distribution(x::Ts; bins = 10, log_scale = false, epochs = time_support(x))
    compute_isi_distribution(g::TsGroup; bins = 10, log_scale = false, epochs = time_support(g))

Histogram of inter-spike intervals (pynapple's `nap.compute_isi_distribution`).
ISIs are computed inside each epoch of `epochs`.

- `bins`: a number of bins spanning the shortest to the longest ISI (of all units, for a
  `TsGroup`), or a vector of bin edges in seconds.
- `log_scale = true`: with an integer `bins`, the bins are log-spaced (equal width in
  log10(ISI)), the usual choice for ISI distributions. Ignored when edges are given;
  pass `exp10.(range(-3, 1; length = 41))` for explicit log-spaced edges.

Returns a NamedTuple `(edges, centers, counts)`: `counts` is a vector for a `Ts` and a
matrix (bins × units, in the order of `ids(g)`) for a `TsGroup`. `centers` are the
arithmetic bin centers, or the geometric ones with `log_scale`. Bins are `[left, right)`,
except the last one which includes its right edge.

```julia
h = compute_isi_distribution(g[id = 185]; bins = 50, log_scale = true)
plot(h.centers, h.counts; xscale = :log10)
```
"""
function compute_isi_distribution(x::Ts; bins = 10, log_scale::Bool = false,
                                  epochs::IntervalSet = x.time_support)
    isi = _isis(x, epochs)
    edges = _isi_edges(bins, [isi], log_scale)
    return (edges = edges, centers = _centers(edges, log_scale && bins isa Integer),
            counts = _histogram(isi, edges))
end

function compute_isi_distribution(g::TsGroup; bins = 10, log_scale::Bool = false,
                                  epochs::IntervalSet = g.time_support)
    isis = [_isis(u, epochs) for u in g.units]
    edges = _isi_edges(bins, isis, log_scale)
    c = zeros(Int, length(edges) - 1, length(g))
    for (j, isi) in enumerate(isis)
        c[:, j] = _histogram(isi, edges)
    end
    return (edges = edges, centers = _centers(edges, log_scale && bins isa Integer), counts = c)
end

_centers(e, geometric) = geometric ? sqrt.(e[1:end-1] .* e[2:end]) : (e[1:end-1] .+ e[2:end]) ./ 2

# --- binning ---------------------------------------------------------------

# For each epoch: (start, number of full bins). Partial bins at the end of an
# epoch are dropped. Returns the layout and the bin centers.
function _bin_layout(ep::IntervalSet, bin_size::Real)
    bin_size > 0 || throw(ArgumentError("bin_size must be positive"))
    layout = Tuple{Float64,Int}[]
    centers = Float64[]
    for (s, e) in ep
        n = floor(Int, (e - s) / bin_size + 1e-9)
        n == 0 && continue
        push!(layout, (s, n))
        append!(centers, s .+ bin_size .* ((1:n) .- 0.5))
    end
    return layout, centers
end

# Bin index of each sample of sorted `t`, following `layout`.
# Calls f(sample_index, global_bin_index) for every sample inside a bin.
# Bins are [left, right), except the last bin of an epoch which includes its right edge.
function _foreach_bin(f, t::AbstractVector{Float64}, layout, bin_size)
    offset = 0
    for (s, n) in layout
        stop = s + n * bin_size
        lo = searchsortedfirst(t, s)
        hi = searchsortedlast(t, stop)
        for k in lo:hi
            b = min(floor(Int, (t[k] - s) / bin_size) + 1, n)
            f(k, offset + b)
        end
        offset += n
    end
end

function _count_one(t, layout, nbins, bin_size)
    c = zeros(Int, nbins)
    _foreach_bin((_, b) -> (c[b] += 1), t, layout, bin_size)
    return c
end

"""
    count(x::Ts, bin_size; ep = time_support(x))
    count(g::TsGroup, bin_size; ep = time_support(g))

Count events in bins of `bin_size` seconds inside each epoch of `ep`.
Returns a `Tsd` stamped at bin centers: a vector for a `Ts`, a matrix
(bins × units, columns named by unit id) for a `TsGroup`.

Bins start at each epoch's start; an incomplete last bin is dropped.

    count(x; ep = time_support(x))

Without `bin_size`, counts events in each epoch of `ep` (stamped at epoch centers).
"""
function Base.count(x::Ts, bin_size::Real; ep::IntervalSet = x.time_support)
    layout, centers = _bin_layout(ep, bin_size)
    c = _count_one(x.t, layout, length(centers), bin_size)
    return Tsd(centers, c, String[], ep, Val(:unchecked))
end

function Base.count(g::TsGroup, bin_size::Real; ep::IntervalSet = g.time_support)
    layout, centers = _bin_layout(ep, bin_size)
    c = zeros(Int, length(centers), length(g))
    for (j, u) in enumerate(g.units)
        c[:, j] = _count_one(u.t, layout, length(centers), bin_size)
    end
    return Tsd(centers, c, string.(g.ids), ep, Val(:unchecked))
end

function _count_epochs(t, ep)
    return [searchsortedlast(t, e) - searchsortedfirst(t, s) + 1 for (s, e) in ep]
end
_epoch_centers(ep) = (ep.start .+ ep.stop) ./ 2

Base.count(x::Ts; ep::IntervalSet = x.time_support) =
    Tsd(_epoch_centers(ep), _count_epochs(x.t, ep), String[], ep, Val(:unchecked))

function Base.count(g::TsGroup; ep::IntervalSet = g.time_support)
    c = reduce(hcat, (_count_epochs(u.t, ep) for u in g.units); init = zeros(Int, length(ep), 0))
    return Tsd(_epoch_centers(ep), c, string.(g.ids), ep, Val(:unchecked))
end

"""
    bin_average(x::Tsd, bin_size; ep = time_support(x))

Average the samples of `x` in bins of `bin_size` seconds (same bins as `count`).
Bins without samples are `NaN`. Works for vector, matrix and N-d data.
"""
function bin_average(x::Tsd, bin_size::Real; ep::IntervalSet = x.time_support)
    layout, centers = _bin_layout(ep, bin_size)
    nb = length(centers)
    tail = size(x.d)[2:end]
    flat = reshape(x.d, length(x.t), :)
    acc = zeros(Float64, nb, size(flat, 2))
    n = zeros(Int, nb)
    _foreach_bin(x.t, layout, bin_size) do k, b
        @views acc[b, :] .+= flat[k, :]
        n[b] += 1
    end
    acc ./= n          # 0 / 0 gives NaN for empty bins
    out = reshape(acc, nb, tail...)
    out = ndims(x.d) == 1 ? vec(out) : out
    return Tsd(centers, out, x.columns, ep, Val(:unchecked))
end

# --- value_from ------------------------------------------------------------

"""
    value_from(x::Ts, data::Tsd)

For each timestamp of `x` (inside both time supports), take the sample of
`data` closest in time. Typical use: the position of the animal at each spike.
"""
function value_from(x::Ts, data::Tsd)
    isempty(data) && throw(ArgumentError("data is empty"))
    sup = intersect(x.time_support, data.time_support)
    t = x.t[_index_in(x.t, sup)]
    td = data.t
    idx = Vector{Int}(undef, length(t))
    for (i, ti) in enumerate(t)
        j = searchsortedfirst(td, ti)
        idx[i] = j > length(td) ? length(td) :
                 j == 1 ? 1 :
                 (ti - td[j-1] <= td[j] - ti ? j - 1 : j)
    end
    return Tsd(t, _rows(data.d, idx), data.columns, sup, Val(:unchecked))
end

# --- threshold -------------------------------------------------------------

"""
    threshold(x::Tsd, thr; method = :above)

Keep the samples of a vector `Tsd` that pass the threshold. `method` is one of
`:above`, `:below`, `:aboveequal`, `:belowequal`.

The new time support covers the runs of passing samples. Its edges are placed
halfway between a passing sample and its non-passing neighbour, or at the
epoch edge when the run touches it. Example: `running = time_support(threshold(speed, 5))`.
"""
function threshold(x::Tsd{<:AbstractVector}, thr::Real; method::Symbol = :above)
    cmp = method === :above ? (>) :
          method === :below ? (<) :
          method === :aboveequal ? (>=) :
          method === :belowequal ? (<=) :
          throw(ArgumentError("method must be :above, :below, :aboveequal or :belowequal"))
    pass = cmp.(x.d, thr)
    t = x.t
    s, e = Float64[], Float64[]
    for (es, ee) in x.time_support
        lo, hi = searchsortedfirst(t, es), searchsortedlast(t, ee)
        i = lo
        while i <= hi
            if pass[i]
                j = i
                while j < hi && pass[j+1]
                    j += 1
                end
                push!(s, i == lo ? es : (t[i-1] + t[i]) / 2)
                push!(e, j == hi ? ee : (t[j] + t[j+1]) / 2)
                i = j + 1
            else
                i += 1
            end
        end
    end
    sup = IntervalSet(s, e)
    keep = findall(pass)
    return Tsd(t[keep], x.d[keep], x.columns, sup, Val(:unchecked))
end

# --- peri-event alignment ---------------------------------------------------

_window(w::Real) = (-Float64(w), Float64(w))
_window(w::Tuple{Real,Real}) = (Float64(w[1]), Float64(w[2]))
_event_times(ev::Ts) = ev.t
_event_times(ev::AbstractVector{<:Real}) = collect(Float64, ev)
_event_times(ev::IntervalSet) = copy(ev.start)

# trial metadata copied from an IntervalSet of trials (fields that clash with the
# perievent fields are prefixed with `trial_`)
_event_metadata(ev) = Dict{Symbol,Vector}()
function _event_metadata(ev::IntervalSet)
    md = Dict{Symbol,Vector}()
    for (k, v) in ev.metadata
        md[k in (:rate, :event_time) ? Symbol(:trial_, k) : k] = copy(v)
    end
    md[:event_stop] = copy(ev.stop)
    return md
end

"""
    compute_perievent(x::Ts, events, window)
    compute_perievent(g::TsGroup, events, window)

Align spikes to each event. `window` is `(before, after)` relative to the event,
e.g. `(-0.5, 1.0)`, or a single number `w` for `(-w, w)`. `events` is

- a vector of times: trials keep the order of the vector, so you can sort the
  events (by trial type, reaction time...) before aligning;
- a `Ts`: trials are in chronological order (a `Ts` is always sorted);
- an `IntervalSet` of trials: events are the trial starts, and every trial metadata
  field (e.g. `:trial_type`) is copied to the result, along with `:event_stop`.

For a `Ts`, returns a `TsGroup` with one trial per event (ids `1:n`, times relative
to the event, time support = the window, metadata `:event_time`).
For a `TsGroup`, returns a vector of such groups, one per unit.

The raster recipe labels the rows of such a group "Trial" (it recognises the
`:event_time` metadata). `plot(pe; yflip = true)` puts the first trial on top.
"""
function compute_perievent(x::Ts, events, window)
    w1, w2 = _window(window)
    w1 < w2 || throw(ArgumentError("window must satisfy before < after"))
    ev = _event_times(events)
    trials = Vector{Vector{Float64}}(undef, length(ev))
    for (i, e) in enumerate(ev)
        lo = searchsortedfirst(x.t, e + w1)
        hi = searchsortedlast(x.t, e + w2)
        # clamp guards against rounding pushing edge spikes outside the window
        trials[i] = clamp.(x.t[lo:hi] .- e, w1, w2)
    end
    return TsGroup(trials; time_support = IntervalSet(w1, w2), event_time = ev,
                   pairs(_event_metadata(events))...)
end

compute_perievent(g::TsGroup, events, window) =
    [compute_perievent(u, events, window) for u in g.units]

# --- flatten ---------------------------------------------------------------

"""
    to_tsd(g::TsGroup)

All spikes of the group in one `Tsd`, sorted in time, whose values are the unit ids.
"""
function to_tsd(g::TsGroup)
    t = reduce(vcat, (u.t for u in g.units); init = Float64[])
    id = reduce(vcat, (fill(g.ids[i], length(u)) for (i, u) in enumerate(g.units)); init = Int[])
    p = sortperm(t)
    return Tsd(t[p], id[p], String[], g.time_support, Val(:unchecked))
end
